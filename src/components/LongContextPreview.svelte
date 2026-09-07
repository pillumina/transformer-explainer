<script lang="ts">
	import { attentionVariant, swaWindowSize, dsaTopK } from '~/store';
	import { tick } from 'svelte';

	export let className: string | undefined = undefined;

	// synthetic long-context pattern: n×n attention map for the current variant.
	// NOT model output — a fixed-seed illustration of how each mask/sparse
	// pattern scales beyond the 6-token demo.
	const N = 64;
	const CELL = 4;
	let open = false;
	let canvas: HTMLCanvasElement;

	function mulberry32(seed: number) {
		return () => {
			seed |= 0;
			seed = (seed + 0x6d2b79f5) | 0;
			let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
			t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
			return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
		};
	}

	function syntheticMatrix(variant: string, w: number, k: number): number[][] {
		const rand = mulberry32(42);
		const scores: number[][] = Array.from({ length: N }, () =>
			Array.from({ length: N }, () => (rand() - 0.5) * 8)
		);
		const out: number[][] = Array.from({ length: N }, () => Array(N).fill(NaN));
		for (let i = 0; i < N; i++) {
			const cand: number[] = [];
			for (let j = 0; j < N; j++) {
				if (j > i) continue; // causal
				if (variant === 'swa' && i - j >= w) continue; // sliding window
				cand.push(j);
			}
			let keep = cand;
			if (variant === 'dsa') {
				keep = cand
					.slice()
					.sort((a, b) => scores[i][b] - scores[i][a])
					.slice(0, Math.min(k, cand.length));
			}
			if (!keep.length) continue;
			const maxV = Math.max(...keep.map((j) => scores[i][j]));
			const exps = keep.map((j) => Math.exp(scores[i][j] - maxV));
			const sum = exps.reduce((a, b) => a + b, 0);
			keep.forEach((j, idx) => (out[i][j] = exps[idx] / sum));
		}
		return out;
	}

	function draw(variant: string, w: number, k: number) {
		if (!canvas) return;
		const ctx2d = canvas.getContext('2d');
		if (!ctx2d) return;
		canvas.width = N * CELL;
		canvas.height = N * CELL;
		const m = syntheticMatrix(variant, w, k);
		for (let i = 0; i < N; i++) {
			for (let j = 0; j < N; j++) {
				const v = m[i][j];
				// sqrt gamma keeps thin softmax tails readable at 4px cells
				ctx2d.fillStyle = Number.isFinite(v)
					? `rgba(124, 58, 237, ${Math.min(1, Math.sqrt(Math.max(0, v)) * 1.6).toFixed(3)})`
					: '#f3f4f6';
				ctx2d.fillRect(j * CELL, i * CELL, CELL - 0.5, CELL - 0.5);
			}
		}
	}

	async function toggle() {
		open = !open;
		if (open) {
			await tick();
			draw($attentionVariant, $swaWindowSize, $dsaTopK);
		}
	}

	$: if (open) draw($attentionVariant, $swaWindowSize, $dsaTopK);

	$: caption =
		$attentionVariant === 'dsa'
			? `synthetic scores · n = ${N} · top-${$dsaTopK}`
			: $attentionVariant === 'swa'
				? `synthetic scores · n = ${N} · window ${$swaWindowSize}`
				: `synthetic scores · n = ${N}`;
</script>

<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
<div class={`lc-preview ${className ?? ''}`}>
	<button class="lc-toggle" on:click={toggle}>
		{open ? 'hide' : 'long context'} · n = {N}
	</button>
	{#if open}
		<div class="lc-panel">
			<canvas bind:this={canvas}></canvas>
			<span class="lc-caption">Illustrative — {caption}</span>
		</div>
	{/if}
</div>

<style lang="scss">
	.lc-preview {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: 0.4rem;

		// anchored to the empty bottom-left corner of the attention step;
		// z above the head-block content so the label is never clipped
		&.lc-corner {
			position: absolute;
			left: 0.75rem;
			bottom: 0.5rem;
			z-index: 60;
		}
	}
	.lc-toggle {
		font-size: 0.65rem;
		color: theme('colors.gray.400');
		border: 1px dashed theme('colors.gray.300');
		border-radius: 999px;
		padding: 0.15rem 0.55rem;
		white-space: nowrap;
		cursor: pointer;
		background: rgba(255, 255, 255, 0.65);

		&:hover {
			color: theme('colors.gray.600');
			border-color: theme('colors.gray.400');
		}
	}
	.lc-panel {
		display: flex;
		flex-direction: column;
		gap: 0.3rem;
		padding: 0.45rem;
		background: white;
		border: 1px solid theme('colors.gray.200');
		border-radius: 8px;
		box-shadow: 0 4px 12px 0 rgb(0 0 0 / 0.08);

		canvas {
			width: 256px;
			height: 256px;
			display: block;
			image-rendering: pixelated;
			border-radius: 4px;
		}
	}
	.lc-caption {
		font-size: 0.6rem;
		color: theme('colors.gray.400');
		white-space: nowrap;
	}
</style>
