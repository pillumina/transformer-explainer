<script lang="ts">
	import {
		blockIdx,
		modelData,
		modelMeta,
		variantMatrices,
		attentionVariant,
		attentionHeadIdxTemp,
		gqaNumKVHeads,
		tokens,
		userId,
		isOnAnimation
	} from '~/store';
	import { tick, onMount } from 'svelte';
	import { kvHeadForQuery, GROUP_COLORS } from '~/utils/attentionVariants';
	import { textPages } from '~/utils/textbookPages';

	export let className: string | undefined = undefined;

	const CELL = 4; // px per matrix cell inside a thumbnail

	$: headList = Array.from({ length: $modelMeta.attention_head_num }, (_, h) => h);
	$: nGroups = $attentionVariant === 'gqa' ? $gqaNumKVHeads : 0;
	$: headsPerGroup = nGroups ? $modelMeta.attention_head_num / nGroups : $modelMeta.attention_head_num;
	$: groupColors =
		$attentionVariant === 'gqa'
			? Array.from({ length: nGroups }, (_, g) => GROUP_COLORS[g % GROUP_COLORS.length])
			: null;
	// the group the currently selected head belongs to — captioned above the grid
	$: activeGroup = Math.floor($attentionHeadIdxTemp / (headsPerGroup || 1));
	$: shareStart = activeGroup * headsPerGroup + 1;
	$: shareEnd = (activeGroup + 1) * headsPerGroup;

	let root: HTMLDivElement;

	function dataFor(
		h: number,
		variant: string,
		bIdx: number,
		md: any,
		vm: any,
		toks: string[]
	): number[][] | null {
		if (variant === 'gqa' || variant === 'swa' || variant === 'dsa') {
			return vm?.[bIdx]?.[h]?.softmaxed ?? null;
		}
		// MHA / MLA: the ONNX graph natively exports every head's softmax
		const raw = md?.outputs?.[`block_${bIdx}_attn_head_${h}_attn_softmax`]?.data;
		if (!raw || !Array.isArray(raw?.[0])) return null;
		let m = raw;
		while (Array.isArray(m?.[0]?.[0])) m = m[0];
		return m as unknown as number[][];
	}

	function drawGrid(
		variant: string,
		bIdx: number,
		md: any,
		vm: any,
		toks: string[]
	) {
		if (!root) return;
		const n = toks.length;
		// live query (document order = head order in both template branches):
		// array bindings proved unreliable across variant re-mounts
		const cells = root.querySelectorAll('canvas');
		// iterate HEADS (not tokens — tokens only size the n×n canvas)
		for (let h = 0; h < Math.min(cells.length, $modelMeta.attention_head_num); h++) {
			const cv = cells[h];
			const ctx2d = cv.getContext('2d');
			if (!ctx2d) continue;
			cv.width = n * CELL;
			cv.height = n * CELL;
			const m = dataFor(h, variant, bIdx, md, vm, toks);
			for (let i = 0; i < n; i++) {
				for (let j = 0; j < n; j++) {
					const v = m?.[i]?.[j];
					// sqrt gamma: near-uniform heads keep a faint but readable tint
					ctx2d.fillStyle = Number.isFinite(v)
						? `rgba(124, 58, 237, ${Math.max(0.05, Math.min(1, Math.sqrt(Math.max(0, v)))).toFixed(3)})`
						: '#e5e7eb';
					ctx2d.fillRect(j * CELL, i * CELL, CELL - 0.5, CELL - 0.5);
				}
			}
		}
	}

	$: if (root) drawGrid($attentionVariant, $blockIdx, $modelData, $variantMatrices, $tokens);
	onMount(async () => {
		// canvases bind after the first reactive pass — redraw once bound
		await tick();
		if (root) drawGrid($attentionVariant, $blockIdx, $modelData, $variantMatrices, $tokens);
	});

	const selectHead = (h: number) => {
		textPages.find((page) => page.id === 'multi-head')?.complete();
		$attentionHeadIdxTemp = h;
		window.dataLayer?.push({
			event: `pagination-attention-head-grid`,
			page_num: h,
			pagination_name: 'attention-head',
			user_id: $userId
		});
	};
</script>

<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-noninteractive-element-interactions a11y-missing-attribute -->
<div
	class={`head-grid ${className ?? ''}`}
	role="group"
	aria-label="All attention heads"
	bind:this={root}
>
	{#if groupColors}
		<!-- caption of the selected head's KV group — lives with the grid so it
			can never collide with the head-block layout -->
		<div class="group-caption" style={`color:${groupColors[activeGroup]}`}>
			<span class="cap-dot" style={`background:${groupColors[activeGroup]}`}></span>
			Query heads {shareStart}–{shareEnd} share KV head {activeGroup + 1} of {nGroups}
		</div>
	{/if}
	{#if groupColors}
		<!-- GQA: wrap each KV group so the colored link under the cells reads
			as "these query heads share this KV head" -->
		{#each groupColors as color, g (color)}
			<div class="group-wrap" style={`--gc:${color}`}>
				<div class="group-cells">
					{#each Array(headsPerGroup) as _, k (k)}
						{@const h = g * headsPerGroup + k}
						<button
							class="head-cell"
							class:active={$attentionHeadIdxTemp === h}
							style={`border-color:${color}`}
							disabled={$isOnAnimation}
							on:click={() => selectHead(h)}
							title={`Head ${h + 1} · KV head ${g + 1}`}
						>
							<canvas></canvas>
						</button>
					{/each}
				</div>
				<div class="group-link"><span>KV {g + 1}</span></div>
			</div>
		{/each}
	{:else}
		{#each headList as h (h)}
			<button
				class="head-cell"
				class:active={$attentionHeadIdxTemp === h}
				disabled={$isOnAnimation}
				on:click={() => selectHead(h)}
				title={`Head ${h + 1}`}
			>
				<canvas></canvas>
			</button>
		{/each}
	{/if}
</div>

<style lang="scss">
	.head-grid {
		position: relative;
		display: flex;
		gap: 4px;
		padding: 3px 3px 2px;
		background: rgba(255, 255, 255, 0.65);
		border-radius: 6px;
		margin-right: 0.4rem;
	}
	// caption of the active KV group, right-aligned above the grid
	.group-caption {
		position: absolute;
		bottom: calc(100% + 5px);
		right: 0;
		display: inline-flex;
		align-items: center;
		gap: 0.3rem;
		font-size: 0.62rem;
		white-space: nowrap;

		.cap-dot {
			width: 0.5rem;
			height: 0.5rem;
			border-radius: 999px;
			flex-shrink: 0;
		}
	}
	.group-wrap {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}
	.group-cells {
		display: flex;
		gap: 2px;
	}
	// the colored "link" under a KV group: ties the query-head cells that
	// share one KV head together, with a tiny KV label
	.group-link {
		position: relative;
		height: 2px;
		border-radius: 999px;
		background: var(--gc);
		opacity: 0.75;

		span {
			position: absolute;
			top: 2px;
			left: 50%;
			transform: translateX(-50%);
			font-size: 0.45rem;
			line-height: 1;
			color: theme('colors.gray.400');
			white-space: nowrap;
		}
	}
	.head-cell {
		position: relative;
		padding: 0;
		line-height: 0;
		background: white;
		border: 1px solid theme('colors.gray.200');
		border-radius: 3px;
		cursor: pointer;
		transition: border-color 0.15s;

		canvas {
			width: 24px;
			height: 24px;
			display: block;
			image-rendering: pixelated;
		}
		&:hover {
			border-color: theme('colors.gray.400');
		}
		&.active {
			border-color: theme('colors.gray.800');
			box-shadow: 0 0 0 1px theme('colors.gray.800');
		}
		&:disabled {
			cursor: default;
		}
	}
</style>
