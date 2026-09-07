<script lang="ts">
	import {
		attentionVariant,
		gqaNumKVHeads,
		swaWindowSize,
		dsaTopK,
		mlaLatentDim,
		modelMeta
	} from '~/store';
	import { kvCacheBytes, type AttentionVariant } from '~/utils/attentionVariants';
	import { Tooltip } from 'flowbite-svelte';

	export let className: string | undefined = undefined;

	const variants: { id: AttentionVariant; label: string }[] = [
		{ id: 'mha', label: 'MHA' },
		{ id: 'gqa', label: 'GQA' },
		{ id: 'swa', label: 'SWA' },
		{ id: 'dsa', label: 'DSA' },
		{ id: 'mla', label: 'MLA' }
	];

	// GQA KV-head presets: must divide the head count
	const kvOptions = [1, 2, 3, 4, 6];
	// MLA latent dimension presets (simulated; DeepSeek V3 uses 512 + 64 RoPE)
	const latentOptions = [128, 256, 512];

	$: variantTitle = {
		mha: 'Multi-Head Attention',
		gqa: 'Grouped-Query Attention',
		swa: 'Sliding-Window Attention',
		dsa: 'DeepSeek Sparse Attention',
		mla: 'Multi-head Latent Attention'
	}[$attentionVariant];

	const CONTEXT_LEN = 1024;
	const formatMB = (bytes: number) => {
		const mb = bytes / (1024 * 1024);
		return mb >= 1 ? `${mb.toFixed(mb >= 10 ? 0 : 1)} MB` : `${(mb * 1024).toFixed(0)} KB`;
	};

	$: mhaCache = kvCacheBytes({
		layers: $modelMeta.layer_num,
		headNum: $modelMeta.attention_head_num,
		dimension: $modelMeta.dimension,
		numKVHeads: $modelMeta.attention_head_num,
		contextLen: CONTEXT_LEN
	});
	$: variantCache = kvCacheBytes({
		layers: $modelMeta.layer_num,
		headNum: $modelMeta.attention_head_num,
		dimension: $modelMeta.dimension,
		// DSA shrinks compute, not the cache: all heads still cache K/V
		numKVHeads: $attentionVariant === 'gqa' ? $gqaNumKVHeads : $modelMeta.attention_head_num,
		contextLen: CONTEXT_LEN,
		slidingWindow: $attentionVariant === 'swa' ? $swaWindowSize : undefined,
		mlaLatentDim: $attentionVariant === 'mla' ? $mlaLatentDim : undefined
	});
	$: cacheFactor = Math.round(mhaCache / variantCache);

	$: cacheNote =
		$attentionVariant === 'swa'
			? 'Rolling buffer keeps only the last w tokens per layer.'
			: $attentionVariant === 'dsa'
				? 'Sparse attention shrinks compute, not the cache — every token is still cached.'
				: $attentionVariant === 'mla'
					? 'Only the shared latent + one RoPE key are cached, not per-head K/V.'
					: 'Fewer KV heads store fewer K/V vectors per token.';

	// ── cache growth curves (context 0 → 1024), MHA baseline vs variant ──
	const CURVE_W = 240;
	const CURVE_H = 96;
	const CURVE_PAD = 10;
	const SAMPLES = 32;
	const sampleCtx = (i: number) => Math.max(1, Math.round((i / SAMPLES) * CONTEXT_LEN));

	$: cacheSeries = (() => {
		const layers = $modelMeta.layer_num;
		const headNum = $modelMeta.attention_head_num;
		const dimension = $modelMeta.dimension;
		const mha: number[] = [];
		const variant: number[] = [];
		for (let i = 0; i <= SAMPLES; i++) {
			const ctx = sampleCtx(i);
			mha.push(
				kvCacheBytes({ layers, headNum, dimension, numKVHeads: headNum, contextLen: ctx })
			);
			variant.push(
				kvCacheBytes({
					layers,
					headNum,
					dimension,
					numKVHeads: $attentionVariant === 'gqa' ? $gqaNumKVHeads : headNum,
					contextLen: ctx,
					slidingWindow: $attentionVariant === 'swa' ? $swaWindowSize : undefined,
					mlaLatentDim: $attentionVariant === 'mla' ? $mlaLatentDim : undefined
				})
			);
		}
		return { mha, variant };
	})();

	// y is normalized against the MHA endpoint (the largest value by design)
	$: yFor = (bytes: number) =>
		CURVE_H - CURVE_PAD - (bytes / cacheSeries.mha[SAMPLES]) * (CURVE_H - CURVE_PAD * 2);
	$: xFor = (i: number) => CURVE_PAD + (i / SAMPLES) * (CURVE_W - CURVE_PAD * 2);
	$: toPath = (arr: number[]) =>
		arr.map((b, i) => `${i ? 'L' : 'M'}${xFor(i).toFixed(1)},${yFor(b).toFixed(1)}`).join(' ');
	$: mhaPath = toPath(cacheSeries.mha);
	$: variantPath = toPath(cacheSeries.variant);
	$: variantArea = `${variantPath} L${(CURVE_W - CURVE_PAD).toFixed(1)},${(CURVE_H - CURVE_PAD).toFixed(1)} L${CURVE_PAD},${(CURVE_H - CURVE_PAD).toFixed(1)} Z`;
	$: variantShortName = { mha: 'MHA', gqa: 'GQA', swa: 'SWA', dsa: 'DSA', mla: 'MLA' }[
		$attentionVariant
	];

	const onClick = (e: Event) => {
		e.stopPropagation();
	};
</script>

<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
<div class={className} on:click={onClick} role="group">
	<div class="controls-row">
		<div class="segmented">
			{#each variants as v (v.id)}
				<button
					class="pill"
					class:active={$attentionVariant === v.id}
					on:click={() => attentionVariant.set(v.id)}
					data-click={`variant-${v.id}`}
				>
					{v.label}
				</button>
			{/each}
		</div>

		{#if $attentionVariant === 'gqa'}
			<div class="param">
				<span class="param-label">KV heads</span>
				<div class="segmented subtle">
					{#each kvOptions as n (n)}
						<button
							class="pill"
							class:active={$gqaNumKVHeads === n}
							on:click={() => gqaNumKVHeads.set(n)}
						>
							{n === 1 ? 'MQA' : n}
						</button>
					{/each}
				</div>
			</div>
		{:else if $attentionVariant === 'swa'}
			<div class="param">
				<span class="param-label">window</span>
				<input
					type="range"
					min="2"
					max="8"
					step="1"
					value={$swaWindowSize}
					on:input={(e) => swaWindowSize.set(Number(e.currentTarget?.value))}
					class="window-slider"
				/>
				<span class="param-value">w = {$swaWindowSize}</span>
			</div>
		{:else if $attentionVariant === 'dsa'}
			<div class="param">
				<span class="param-label">select top-k</span>
				<div class="segmented subtle">
					{#each [2, 3, 4, 6, 8] as n (n)}
						<button class="pill" class:active={$dsaTopK === n} on:click={() => dsaTopK.set(n)}>
							{n}
						</button>
					{/each}
				</div>
			</div>
		{:else if $attentionVariant === 'mla'}
			<div class="param">
				<span class="param-label">latent dim</span>
				<div class="segmented subtle">
					{#each latentOptions as n (n)}
						<button
							class="pill"
							class:active={$mlaLatentDim === n}
							on:click={() => mlaLatentDim.set(n)}
						>
							{n}
						</button>
					{/each}
				</div>
			</div>
		{/if}

		{#if $attentionVariant !== 'mha'}
			<!-- badge sits BEFORE the wide cache chip: keeps its right edge well
				clear of the "MLP" step title at common viewport widths -->
			<Tooltip class="popover" placement="bottom" triggeredBy=".variant-badge">
				{$attentionVariant === 'dsa'
					? 'Indexer scores and top-k selection are recomputed in-browser from the real GPT-2 attention scores (the trained indexer approximates query-key relevance) — the select-then-attend structure matches DeepSeek Sparse Attention.'
					: $attentionVariant === 'mla'
						? 'The latent compression is illustrative: with MLA, up-projecting K/V from the shared latent is absorbable into Q, so attention scores are unchanged — we show the real GPT-2 scores and visualize the latent pipeline + cache savings.'
						: 'Attention matrices are recomputed in-browser from the real GPT-2 Q / K / V projections — only the head grouping (GQA) or the mask pattern (SWA) differs from native GPT-2.'}
			</Tooltip>
			<span class="variant-badge">
				{$attentionVariant === 'mla' ? 'Illustrative pipeline' : 'Simulated on GPT-2'}
			</span>

			<Tooltip class="popover cache-popover" triggeredBy=".cache-chip" placement="bottom">
				<div class="cache-detail">
					<svg
						class="cache-curve"
						viewBox={`0 0 ${CURVE_W} ${CURVE_H}`}
						role="img"
						aria-label="KV cache growth: MHA vs {$attentionVariant.toUpperCase()}"
					>
						<line
							x1={CURVE_PAD}
							y1={CURVE_H - CURVE_PAD}
							x2={CURVE_W - CURVE_PAD}
							y2={CURVE_H - CURVE_PAD}
							class="curve-axis"
						/>
						<path d={variantArea} class="curve-area" />
						<path d={mhaPath} class="curve-mha" />
						<path d={variantPath} class="curve-variant" />
						<circle cx={xFor(SAMPLES)} cy={yFor(mhaCache)} r="2.6" class="dot-mha" />
						<circle cx={xFor(SAMPLES)} cy={yFor(variantCache)} r="2.6" class="dot-variant" />
					</svg>
					<div class="curve-legend">
						<span class="legend-mha">MHA · {formatMB(mhaCache)}</span>
						<span class="legend-variant">{variantShortName} · {formatMB(variantCache)}</span>
					</div>
					<span class="cache-note">
						KV cache growth over a {CONTEXT_LEN}-token context (fp32). {cacheNote}
					</span>
				</div>
			</Tooltip>
			<button class="cache-chip" type="button" data-click="cache-chip">
				KV cache
				<b>{cacheFactor > 1 ? `${cacheFactor}× smaller` : 'same size'}</b>
				<svg class="chip-spark" viewBox={`0 0 ${CURVE_W} ${CURVE_H}`} aria-hidden="true">
					<path d={mhaPath} class="curve-mha" />
					<path d={variantPath} class="curve-variant" />
				</svg>
			</button>
		{/if}
	</div>
</div>

<style lang="scss">
	.controls-row {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 0.6rem;
		flex-wrap: nowrap;
		white-space: nowrap;
		// the step title is bottom-anchored with 2rem padding (see +page.svelte);
		// pull the row slightly lower so it never intersects the
		// "Transformer Block N" navigation above (z-index 400 covers it otherwise)
		margin-bottom: -0.7rem;
	}
	.segmented {
		display: inline-flex;
		background: theme('colors.gray.100');
		border: 1px solid theme('colors.gray.200');
		border-radius: 999px;
		padding: 2px;
		gap: 2px;

		&.subtle {
			background: theme('colors.gray.50');
		}
	}
	.pill {
		font-size: 0.72rem;
		line-height: 1;
		padding: 0.28rem 0.6rem;
		border-radius: 999px;
		color: theme('colors.gray.500');
		transition:
			color 0.15s,
			background-color 0.15s,
			box-shadow 0.15s;

		&:hover {
			color: theme('colors.gray.700');
		}
		&.active {
			background: white;
			color: theme('colors.gray.800');
			font-weight: 600;
			box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.06);
			border: 1px solid theme('colors.gray.200');
			padding: calc(0.28rem - 1px) calc(0.6rem - 1px);
		}
	}
	.param {
		display: inline-flex;
		align-items: center;
		gap: 0.4rem;
	}
	.param-label {
		font-size: 0.72rem;
		color: theme('colors.gray.500');
	}
	.param-value {
		font-family: monospace;
		font-size: 0.72rem;
		color: theme('colors.gray.700');
		min-width: 2.6rem;
	}
	.window-slider {
		width: 6rem;
		height: 0.25rem;
		accent-color: theme('colors.purple.600');
		cursor: pointer;
	}
	.variant-badge {
		font-size: 0.68rem;
		color: theme('colors.gray.400');
		border: 1px dashed theme('colors.gray.300');
		border-radius: 999px;
		padding: 0.15rem 0.55rem;
		white-space: nowrap;
	}

	.cache-chip {
		display: inline-flex;
		align-items: center;
		gap: 0.3rem;
		font-size: 0.68rem;
		color: theme('colors.gray.500');
		background: theme('colors.gray.100');
		border: 1px solid theme('colors.gray.200');
		border-radius: 999px;
		padding: 0.18rem 0.55rem;
		white-space: nowrap;
		cursor: help;

		b {
			color: theme('colors.purple.600');
			font-weight: 600;
		}
		&:hover {
			color: theme('colors.gray.700');
			border-color: theme('colors.gray.300');
		}
	}

	.cache-detail {
		display: flex;
		flex-direction: column;
		gap: 0.35rem;
		text-align: left;
	}
	.cache-curve {
		width: 15rem;
		height: auto;
		display: block;
	}
	.curve-axis {
		stroke: theme('colors.gray.200');
		stroke-width: 1;
	}
	.curve-mha {
		fill: none;
		stroke: theme('colors.gray.400');
		stroke-width: 1.5;
		stroke-dasharray: 4 3;
	}
	.curve-variant {
		fill: none;
		stroke: theme('colors.purple.600');
		stroke-width: 2;
	}
	.curve-area {
		fill: theme('colors.purple.100');
		stroke: none;
		opacity: 0.7;
	}
	.dot-mha {
		fill: theme('colors.gray.400');
	}
	.dot-variant {
		fill: theme('colors.purple.600');
	}
	.curve-legend {
		display: flex;
		justify-content: space-between;
		gap: 0.6rem;
		font-size: 0.65rem;
		font-family: monospace;
		color: theme('colors.gray.500');
		.legend-mha {
			color: theme('colors.gray.400');
		}
		.legend-variant {
			color: theme('colors.purple.600');
			font-weight: 600;
		}
	}
	.chip-spark {
		width: 2.75rem;
		height: 0.85rem;
		display: block;
		overflow: visible;
	}
	.cache-note {
		font-size: 0.65rem;
		color: theme('colors.gray.400');
		white-space: normal;
		max-width: 20rem;
	}

	@media (max-width: 1280px) {
		// keep the title to a single line on narrow viewports:
		// drop the explanatory badge, keep the interactive controls
		.variant-badge {
			display: none;
		}
	}
	@media (max-width: 1024px) {
		.cache-chip {
			display: none;
		}
	}
</style>
