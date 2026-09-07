/**
 * Attention-variant math.
 *
 * MHA      → tensors come straight from the ONNX graph (no re-computation).
 * GQA / SWA / DSA → computed here in plain JS from per-head Q / K^T activations
 *             exported by the modified ONNX graph, so every displayed number
 *             is derived from the real model weights.
 * MLA      → the attention *matrices* are mathematically identical to standard
 *             attention (up-projecting K/V from a shared latent is absorbable
 *             into Q), so we reuse the ONNX tensors; the latent compression
 *             lives purely in the pipeline visual + KV-cache accounting.
 *
 * GQA simulation (exact attention math, real GPT-2 projections):
 *   Query head h uses KV head g(h) = floor(h / (H / n_kv)).
 *   scores_h = Q_h · K_{g(h)}^T / sqrt(d_head), then causal mask + softmax.
 *   (Only the attention *matrices* are GQA-exact; GPT-2's c_proj was trained
 *   for MHA, which is why the UI labels this "Simulated on GPT-2".)
 *
 * SWA (Mistral semantics, window w includes the current token):
 *   token i attends to j  iff  j <= i  and  i - j < w.
 *
 * DSA (DeepSeek Sparse Attention, simulated indexer):
 *   A lightweight indexer scores every causal (i, j) pair; here we reuse the
 *   real scaled Q·K^T as the indexer score (the trained indexer approximates
 *   query-key relevance). Row i keeps its top-k scores among j <= i (self is
 *   always kept), everything else is masked before softmax — exactly the
 *   "select then attend" structure of DSA.
 */

export type AttentionVariant = 'mha' | 'gqa' | 'swa' | 'dsa' | 'mla';

/** per-KV-group accent colors used by the GQA visuals (cycled) */
export const GROUP_COLORS = [
	'#2563eb', // blue-600
	'#ea580c', // orange-600
	'#059669', // emerald-600
	'#7c3aed', // violet-600
	'#db2777', // pink-600
	'#0d9488' // teal-600
];

export type VariantMatrices = {
	/** raw dot products Q·K^T (row = query token, col = key token) */
	queryKey: MatrixData;
	/** scaled (÷√d_head) with disallowed positions set to -Infinity */
	masked: MatrixData;
	/** softmax probabilities; disallowed positions set to -Infinity (renders gray) */
	softmaxed: MatrixData;
	/** GQA only: KV head serving this query head */
	kvHeadIdx?: number;
	/** DSA only: positions the indexer selected for each query row */
	selected?: boolean[][];
};

export const softmaxStableRows = (rows: number[][]): number[][] => {
	return rows.map((row) => {
		let max = -Infinity;
		for (const v of row) if (Number.isFinite(v) && v > max) max = v;
		if (!Number.isFinite(max)) return row.map(() => 0);
		const exps = row.map((v) => (Number.isFinite(v) ? Math.exp(v - max) : 0));
		let sum = 0;
		for (const e of exps) sum += e;
		return exps.map((e) => e / sum);
	});
};

export const matmul = (a: number[][], b: number[][]): number[][] => {
	const m = a.length;
	const k = b.length;
	const n = b[0].length;
	const out: number[][] = Array.from({ length: m }, () => new Array(n).fill(0));
	for (let i = 0; i < m; i++) {
		const ai = a[i];
		const oi = out[i];
		for (let p = 0; p < k; p++) {
			const aip = ai[p];
			if (aip === 0) continue;
			const bp = b[p];
			for (let j = 0; j < n; j++) oi[j] += aip * bp[j];
		}
	}
	return out;
};

/** token i is allowed to attend token j (causal) */
const causalAllowed = (i: number, j: number) => j <= i;

/** token i is allowed to attend token j under sliding window w (incl. self) */
const windowAllowed = (i: number, j: number, w: number) => j <= i && i - j < w;

/**
 * DSA indexer selection: for query row i, pick the top-k causal positions j<=i
 * by indexer score (self j=i is always kept — DeepSeek V3 forces this). When
 * fewer than k causal candidates exist, the whole causal row is kept.
 */
const dsaSelect = (scaled: MatrixData, k: number): boolean[][] => {
	return scaled.map((row, i) => {
		const selected: boolean[] = new Array(row.length).fill(false);
		// causal candidates sorted by score, highest first; self (j=i) is free
		const candidates: number[] = [];
		for (let j = 0; j <= i; j++) candidates.push(j);
		candidates.sort((a, b) => row[b] - row[a]);
		const budget = Math.min(k, candidates.length);
		selected[i] = true; // self always attends (also naturally top-ranked)
		let picked = 1; // self counts toward the budget
		for (const j of candidates) {
			if (picked >= budget) break;
			if (j === i) continue;
			selected[j] = true;
			picked++;
		}
		return selected;
	});
};

const applySelectedMask = (rows: MatrixData, selected: boolean[][]): MatrixData =>
	rows.map((row, i) => row.map((v, j) => (selected[i][j] ? v : -Infinity)));

const applyMask = (rows: number[][], allowed: (i: number, j: number) => boolean): MatrixData =>
	rows.map((row, i) => row.map((v, j) => (allowed(i, j) ? v : -Infinity)));

export const kvHeadForQuery = (queryHeadIdx: number, headNum: number, numKVHeads: number) =>
	Math.floor(queryHeadIdx / (headNum / numKVHeads));

/** strips any leading singleton dims: [1,S,64] → [S,64], [1,S,S] → [S,S] */
const asMatrix = (d: unknown): number[][] | null => {
	if (!Array.isArray(d) || !Array.isArray(d[0])) return null;
	let m = d as unknown[] as number[][][];
	while (Array.isArray(m?.[0]?.[0])) m = m[0] as unknown as number[][][];
	return m as unknown as number[][];
};

const out = (key: string, modelData: ModelData) => {
	const data = modelData?.outputs?.[key]?.data;
	return data ? asMatrix(data) : null;
};

/**
 * Computes the per-head matrices for GQA / SWA / DSA. MLA needs no per-head
 * math (returns null → ONNX tensors used directly). Returns null whenever the
 * required activations are unavailable (e.g. cached example data or MHA/MLA).
 */
export const computeVariantMatrices = ({
	modelData,
	blockIdx,
	headIdx,
	headNum,
	dimension,
	variant,
	numKVHeads,
	windowSize,
	dsaTopK
}: {
	modelData: ModelData;
	blockIdx: number;
	headIdx: number;
	headNum: number;
	dimension: number;
	variant: AttentionVariant;
	numKVHeads: number;
	windowSize: number;
	dsaTopK: number;
}): VariantMatrices | null => {
	if (variant === 'mha' || variant === 'mla') return null;

	const dHead = dimension / headNum;

	if (variant === 'gqa') {
		const kvHeadIdx = kvHeadForQuery(headIdx, headNum, numKVHeads);
		const q = out(`block_${blockIdx}_attn_head_${headIdx}_q`, modelData);
		const kt = out(`block_${blockIdx}_attn_head_${kvHeadIdx}_k_transposed`, modelData);
		if (!q || !kt) return null;

		// q: [S, d_head] · kt: [d_head, S] → raw scores [S, S]
		const queryKey = matmul(q, kt);
		const masked = applyMask(queryKey, causalAllowed).map((row) =>
			row.map((v) => (Number.isFinite(v) ? v / Math.sqrt(dHead) : v))
		);
		const softmaxed = applyMask(softmaxStableRows(masked), causalAllowed);
		return { queryKey, masked, softmaxed, kvHeadIdx };
	}

	if (variant === 'dsa') {
		// indexer score ≈ real scaled QK^T (see header note); then top-k select
		const raw = out(`block_${blockIdx}_attn_head_${headIdx}_attn`, modelData);
		const scaled = out(`block_${blockIdx}_attn_head_${headIdx}_attn_scaled`, modelData);
		if (!raw || !scaled) return null;

		const selected = dsaSelect(scaled, dsaTopK);
		const masked = applySelectedMask(scaled, selected);
		const softmaxed = applyMask(softmaxStableRows(masked), causalAllowed);
		return { queryKey: raw, masked, softmaxed, selected };
	}

	// SWA — reuses the ONNX-provided QK^T tensors (raw and pre-mask scaled).
	const raw = out(`block_${blockIdx}_attn_head_${headIdx}_attn`, modelData);
	const scaled = out(`block_${blockIdx}_attn_head_${headIdx}_attn_scaled`, modelData);
	if (!raw || !scaled) return null;

	const allowed = (i: number, j: number) => windowAllowed(i, j, windowSize);
	const masked = applyMask(scaled, allowed);
	const softmaxed = applyMask(softmaxStableRows(masked), allowed);
	return { queryKey: raw, masked, softmaxed };
};

/**
 * Dev-only sanity check: recompute standard MHA from the exported per-head
 * Q / K^T and compare against the ONNX graph's own scaled + softmax outputs.
 * Exposed via `window.__teVerify` in dev mode.
 */
export const verifyMhaAgainstOnnx = (
	modelData: ModelData,
	blockIdx: number,
	headNum: number,
	dimension: number
) => {
	const report: { head: number; maxScaledDiff: number; maxSoftmaxDiff: number }[] = [];
	for (let h = 0; h < headNum; h++) {
		const q = out(`block_${blockIdx}_attn_head_${h}_q`, modelData);
		const kt = out(`block_${blockIdx}_attn_head_${h}_k_transposed`, modelData);
		const onnxScaled = out(`block_${blockIdx}_attn_head_${h}_attn_scaled`, modelData);
		const onnxSoftmax = out(`block_${blockIdx}_attn_head_${h}_attn_softmax`, modelData);
		if (!q || !kt || !onnxScaled || !onnxSoftmax) return null;

		const raw = matmul(q, kt);
		const scaled = raw.map((row) => row.map((v) => v / Math.sqrt(dimension / headNum)));
		const softmaxed = softmaxStableRows(applyMask(scaled, causalAllowed));

		let maxScaledDiff = 0;
		let maxSoftmaxDiff = 0;
		for (let i = 0; i < scaled.length; i++) {
			for (let j = 0; j <= i; j++) {
				maxScaledDiff = Math.max(maxScaledDiff, Math.abs(scaled[i][j] - onnxScaled[i][j]));
				maxSoftmaxDiff = Math.max(maxSoftmaxDiff, Math.abs(softmaxed[i][j] - onnxSoftmax[i][j]));
			}
		}
		report.push({ head: h, maxScaledDiff, maxSoftmaxDiff });
	}
	return report;
};

/**
 * KV-cache footprint (bytes) after `contextLen` tokens, fp32.
 * MHA/GQA: cache grows linearly with context; n_kv heads of K+V per layer.
 * SWA: rolling buffer keeps only the last `slidingWindow` tokens per layer
 *      (for all heads), so the cache is bounded regardless of context length.
 * MLA: one shared latent vector (mlaLatentDim) + one decoupled RoPE key
 *      (d_head) per token per layer — independent of the number of heads.
 */
export const kvCacheBytes = ({
	layers,
	headNum,
	dimension,
	numKVHeads,
	contextLen,
	slidingWindow,
	mlaLatentDim
}: {
	layers: number;
	headNum: number;
	dimension: number;
	numKVHeads: number;
	contextLen: number;
	slidingWindow?: number;
	mlaLatentDim?: number;
}) => {
	const dHead = dimension / headNum;
	if (mlaLatentDim) {
		return contextLen * layers * (mlaLatentDim + dHead) * 4;
	}
	if (slidingWindow) {
		return layers * Math.min(slidingWindow, contextLen) * headNum * dHead * 2 * 4;
	}
	return contextLen * layers * numKVHeads * dHead * 2 * 4;
};
