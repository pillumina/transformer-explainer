<script lang="ts">
	import {
		expandedBlock,
		tokens,
		modelData,
		rootRem,
		attentionHeadIdx,
		hoveredMatrixCell,
		blockIdx,
		isExpandOrCollapseRunning,
		userId,
		variantMatrices,
		attentionVariant,
		dsaTopK,
		gqaNumKVHeads,
		swaWindowSize
	} from '~/store';
	import classNames from 'classnames';
	import Matrix from '~/components/common/Matrix.svelte';
	import { gsap } from '~/utils/gsap';
	import { maskArray } from '~/utils/array';
	import { getContext, onMount, tick } from 'svelte';
	import resolveConfig from 'tailwindcss/resolveConfig';
	import tailwindConfig from '../../tailwind.config';
	import * as d3 from 'd3';
	import Katex from '~/utils/Katex.svelte';
	import { Tooltip } from 'flowbite-svelte';
	import { ATTENTION_OUT } from '~/constants/opacity';
	import { ga } from '~/utils/event';
	import { ZoomInOutline } from 'flowbite-svelte-icons';
	import TextbookTooltip from '~/components/common/TextbookTooltip.svelte';
	import { textPages } from '~/utils/textbookPages';
	import { highlightAttentionPath, removeAttentionPathHighlight } from '~/utils/textbook';

	const { theme } = resolveConfig(tailwindConfig);

	$: placeHolderData = Array($tokens.length)
		.fill(0)
		.map((col) => Array($tokens.length).fill(-Infinity));
	// MHA/MLA: ONNX tensors used directly (MLA's latent up-projection is
	// absorbable into Q, so the attention matrices are identical to MHA).
	// GQA/SWA/DSA: client-side computed matrices take precedence, and we never
	// fall back to the MHA tensors (that would silently display MHA data) —
	// show placeholders until the activations are available.
	$: useOnnx = $attentionVariant === 'mha' || $attentionVariant === 'mla';
	$: vm = $variantMatrices?.[$blockIdx]?.[$attentionHeadIdx] || null;
	$: queryKey =
		vm?.queryKey ||
		(useOnnx
			? $modelData?.outputs?.[`block_${$blockIdx}_attn_head_${$attentionHeadIdx}_attn`]?.data
			: null) ||
		placeHolderData;
	$: masked =
		vm?.masked ||
		(useOnnx
			? $modelData?.outputs?.[`block_${$blockIdx}_attn_head_${$attentionHeadIdx}_attn_masked`]?.data
			: null) ||
		placeHolderData;
	$: softmaxed =
		vm?.softmaxed ||
		(useOnnx
			? $modelData?.outputs?.[`block_${$blockIdx}_attn_head_${$attentionHeadIdx}_attn_dropout`]
					?.data
			: null) ||
		placeHolderData;

	/**
	 * Re-mount the matrices whenever what they display can change. Reactive prop
	 * updates alone proved unreliable across variant switches (a matrix could
	 * keep drawing data from the era it mounted in), so key on the live stores —
	 * head changes intentionally stay OUT of the key: they redraw in place, the
	 * way the original site animates head switching.
	 */
	$: matrixKey = [
		$attentionVariant,
		$blockIdx,
		$dsaTopK,
		$gqaNumKVHeads,
		$swaWindowSize,
		$tokens.length
	].join('|');
	let renderedKey = '';
	$: if (matrixKey !== renderedKey) {
		const remounted = renderedKey !== '';
		renderedKey = matrixKey;
		if (remounted && isAttentionExpanded) {
			// matrices re-created mid-expansion: reapply the expanded end-state,
			// otherwise the re-mounted layers show their markup opacities
			tick().then(() => {
				const setOp = (root: HTMLElement | undefined, sel: string, op: string) =>
					root?.querySelector(sel)?.style.setProperty('opacity', op);
				setOp(attentionMask, '.prev', '0');
				setOp(attentionMask, '.main', '1');
				setOp(attentionSoftmax, '.prev', '0');
				setOp(attentionSoftmax, '.main', '1');
				if ($attentionVariant === 'dsa' || $attentionVariant === 'swa') {
					// keep the indexer/window's rejected cells dimmed, matching the
					// end state of the expansion animation
					const svg = attentionQK?.querySelector('.main.matrix-container svg.matrix-svg');
					const circles = svg?.querySelectorAll('circle');
					const cols = masked[0]?.length ?? 0;
					if (circles && cols && circles.length === masked.length * cols) {
						circles.forEach((c, idx) => {
							const i = Math.floor(idx / cols);
							const j = idx % cols;
							if (masked[i] && !Number.isFinite(masked[i][j])) c.style.opacity = '0.25';
						});
					}
				}
			});
		}
	}

	let factor = 1; //todo
	let maxCellSize = 20 * factor;
	let minCellSize = 10 * factor;
	$: cellSize = Math.min(
		maxCellSize,
		Math.max((1 / $tokens.length) * rootRem * 6 * factor, minCellSize)
	);

	let attentionQK: HTMLDivElement;
	let attentionMask: HTMLDivElement;
	let attentionSoftmax: HTMLDivElement;
	let attentionResult: HTMLDivElement;

	let attentionMatrixWidth = 0;

	let isAttentionExpanded = false;

	const blockId = getContext('block-id');

	// event handling

	$: if ($expandedBlock.id !== blockId && isAttentionExpanded) {
		isAttentionExpanded = false;
		collapseAttention();
	}
	$: if ($expandedBlock.id === blockId && !isAttentionExpanded) {
		isAttentionExpanded = true;
		expandAttention();
	}

	const onClickAttention = (e) => {
		e.stopPropagation();
		e.preventDefault();
		textPages.find((page) => page.id === 'masked-self-attention')?.complete();

		if (!isAttentionExpanded) {
			expandedBlock.set({ id: blockId });
		}
	};

	let expandableEl: HTMLDivElement;

	function handleOutsideClick(e) {
		if (!isAttentionExpanded) return;
		// the head-grid thumbnails / long-context preview live outside the
		// matrices but operate ON the expanded view — clicking them must not
		// collapse it
		const t = e.target as Element;
		if (t?.closest?.('.head-grid, .lc-corner')) return;
		if (!expandableEl.contains(e.target)) {
			expandedBlock.set({ id: null });
		}
	}
	onMount(() => {
		document.querySelector('.main-section').addEventListener('click', handleOutsideClick);
		return () => {
			document.querySelector('.main-section').removeEventListener('click', handleOutsideClick);
		};
	});

	// animation
	let expandTl = gsap.timeline();
	let collapseTl = gsap.timeline();

	// google analytics
	let startTime = null;

	const expandAttention = () => {
		highlightAttentionPath();

		isAttentionExpanded = true;
		isExpandOrCollapseRunning.set(true);
		collapseTl.progress(1);

		const keyPaths = document.querySelectorAll('div.sankey g.attention path.key-to-attention');
		const queryPaths = document.querySelectorAll('div.sankey g.attention path.query-to-attention');
		const outPaths = document.querySelectorAll('div.sankey g.attention path.to-attention-out');

		[...keyPaths, ...queryPaths].forEach((path) => {
			const length = path.getTotalLength();
			path.style.strokeDasharray = length;
			path.style.strokeDashoffset = length;
		});

		const QKDuration = 1.2;
		const stagger = Number((QKDuration / $tokens.length).toFixed(2));

		// MLA: the QK matrix visually expands out of the narrow c_KV latent —
		// start it squeezed horizontally and grow to full width
		gsap.set(attentionQK, {
			scaleX: $attentionVariant === 'mla' ? 0.35 : 1,
			transformOrigin: '50% 50%'
		});

		expandTl
			.set([attentionMask.querySelector('.prev'), attentionSoftmax.querySelector('.prev')], {
				opacity: 1
			})
			.set([attentionMask.querySelector('.main'), attentionSoftmax.querySelector('.main')], {
				opacity: 0
			});
		expandTl.set(outPaths, { opacity: 0 });

		expandTl.to(attentionResult, {
			opacity: 0,
			display: 'none',
			duration: 0.2
		});

		// show QK
		expandTl
			.set(attentionQK, {
				width: 'auto',
				display: 'flex',
				opacity: 0
			})
			.to(attentionQK, {
				opacity: 1,
				duration: 0.5
			})
			.to(keyPaths, {
				strokeDashoffset: 0,
				stagger,
				duration: QKDuration,
				ease: 'power2.out'
				// ease: 'back.out(1.7)'
			})
			.to(
				queryPaths,
				{
					strokeDashoffset: 0,
					stagger,
					duration: QKDuration,
					// ease: 'back.out(1.7)'
					ease: 'power2.out'
				},
				'<'
			)
			.from(
				attentionQK.querySelectorAll('svg circle'),
				{
					scale: 0,
					transformOrigin: '50% 50%',
					opacity: 0,
					delay: QKDuration / $tokens.length,
					stagger: Number((QKDuration / Math.pow($tokens.length, 2)).toFixed(2)),
					ease: 'power2.out',
					// ease: 'back.out(1.7)',
					duration: QKDuration
				},
				'<'
			);

		// MLA: finish the "up-projected from c_KV" expansion
		if ($attentionVariant === 'mla') {
			expandTl.to(
				attentionQK,
				{
					scaleX: 1,
					duration: QKDuration,
					ease: 'power2.out'
				},
				'<'
			);
		}

		// DSA / SWA: pre-dim the cells the indexer/window REJECTS while the
		// scores are still on screen — the mask panel's grey cells then read
		// as "the same positions, now masked out" instead of new information
		if ($attentionVariant === 'dsa' || $attentionVariant === 'swa') {
			const qkSvg = attentionQK.querySelector('.main.matrix-container svg.matrix-svg');
			if (qkSvg) {
				const qkCircles = qkSvg.querySelectorAll('circle');
				const cols = masked[0]?.length ?? 0;
				if (cols && qkCircles.length === masked.length * cols) {
					const rejected: Element[] = [];
					qkCircles.forEach((c, idx) => {
						const i = Math.floor(idx / cols);
						const j = idx % cols;
						if (masked[i] && !Number.isFinite(masked[i][j])) rejected.push(c);
					});
					if (rejected.length) {
						expandTl.to(
							rejected,
							{
								opacity: 0.25,
								duration: 0.5,
								stagger: 0.015,
								ease: 'power2.out',
								// the pop-in .from may still be running for late circles;
								// without overwrite it would write opacity back to 1
								overwrite: 'auto'
							},
							'<+1.4'
						);
					}
				}
			}
		}

		// show Masked
		expandTl
			.set(attentionMask, { width: 0, x: attentionMatrixWidth * -1, opacity: 0 })
			.to(attentionMask, {
				opacity: 1,
				display: 'flex',
				width: attentionMatrixWidth,
				x: 0,
				duration: 0.5
			})
			.to(attentionMask.querySelector('.prev'), {
				opacity: 0,
				duration: 1
			})
			.to(
				attentionMask.querySelector('.main'),
				{
					opacity: 1,
					duration: 1
				},
				'<'
			);

		// DSA / SWA: greyed-out (disallowed) cells pop in row by row while the
		// scores fade — makes the indexer/window "filtering" visible
		if ($attentionVariant === 'dsa' || $attentionVariant === 'swa') {
			const grayCircles = getGrayCircles(attentionMask, masked);
			if (grayCircles.length) {
				expandTl.from(
					grayCircles,
					{
						scale: 0.3,
						opacity: 0.1,
						transformOrigin: '50% 50%',
						duration: 0.3,
						stagger: 0.015,
						ease: 'power2.out'
					},
					'<+0.3'
				);
			}
		}

		// SWA: ripple the kept window band row by row — after the mask is
		// applied, the surviving cells pulse top to bottom, which reads as
		// the window sliding down the diagonal
		if ($attentionVariant === 'swa') {
			const bandSvg = attentionMask.querySelector('.main.matrix-container svg.matrix-svg');
			if (bandSvg) {
				const bandCircles = bandSvg.querySelectorAll('circle');
				const cols = masked[0]?.length ?? 0;
				if (cols && bandCircles.length === masked.length * cols) {
					const kept: Element[] = [];
					bandCircles.forEach((c, idx) => {
						const i = Math.floor(idx / cols);
						const j = idx % cols;
						if (masked[i] && Number.isFinite(masked[i][j])) kept.push(c);
					});
					if (kept.length) {
						expandTl.to(
							kept,
							{
								scale: 1.3,
								transformOrigin: '50% 50%',
								duration: 0.18,
								ease: 'power2.out',
								yoyo: true,
								repeat: 1,
								stagger: { each: 0.05 }
							},
							'<+1.1'
						);
					}
				}
			}
		}

		// show Softmaxed
		expandTl
			.set(attentionSoftmax, { width: 0, x: attentionMatrixWidth * -1, opacity: 0 })
			.to(attentionSoftmax, {
				opacity: 1,
				display: 'flex',
				width: attentionMatrixWidth,
				x: 0,
				duration: 0.4
			})
			.to(attentionSoftmax.querySelector('.prev'), {
				opacity: 0,
				duration: 1
			})
			.to(
				attentionSoftmax.querySelector('.main'),
				{
					opacity: 1,
					duration: 1
				},
				'<'
			);

		expandTl.to(outPaths, {
			opacity: ATTENTION_OUT,
			onComplete: () => {
				isExpandOrCollapseRunning.set(false);
			}
		});

		startTime = performance.now();
		window.dataLayer?.push({
			event: 'visibility-show',
			visible_name: 'attention-expansion',
			start_time: startTime,
			user_id: $userId
		});
	};

	const collapseAttention = () => {
		removeAttentionPathHighlight();
		let endTime = performance.now();
		let visibleDuration = endTime - startTime;

		window.dataLayer?.push({
			event: 'visibility-hide',
			visible_name: 'attention-expansion',
			end_time: endTime,
			visible_duration: visibleDuration,
			user_id: $userId
		});

		isAttentionExpanded = false;
		isExpandOrCollapseRunning.set(true);
		expandTl.progress(1);
		collapseTl.to([attentionQK, attentionMask, attentionSoftmax], {
			opacity: 0,
			display: 'none',
			width: 0,
			duration: 0.5
		});

		collapseTl.to(
			attentionResult,
			{
				opacity: 1,
				display: 'flex',
				duration: 0.5,
				onComplete: () => {
					isExpandOrCollapseRunning.set(false);
				}
			},
			0
		);
	};

	// color scale
	// extent over finite values only: placeholder data (all -Infinity) or masked
	// cells must never leak ±Infinity into the legend or the scale domain
	$: qkFiniteExtent = finiteExtent(queryKey);
	$: qkColorScaleDomain = qkFiniteExtent ?? [0, 1];
	$: hasQkData = qkFiniteExtent !== null;
	$: dsaTopKMath = `keep top-${$dsaTopK} indexer scores per row`;

	function finiteExtent(m: MatrixData) {
		let lo = Infinity;
		let hi = -Infinity;
		for (const row of m) {
			for (const x of row) {
				if (Number.isFinite(x)) {
					if (x < lo) lo = x;
					if (x > hi) hi = x;
				}
			}
		}
		return Number.isFinite(lo) ? [lo, hi] : null;
	}

	/**
	 * Circles rendered for disallowed (-Infinity) cells, in row-major order —
	 * used by the DSA/SWA expand animation to "sweep" the masked positions.
	 *
	 * Scope carefully: a page-level ancestor (`.block-steps.main`) makes the
	 * selector '.main svg circle' also match the sibling .prev layer and any
	 * other matrix on the page — so query the one svg inside this matrix's
	 * own `.main` layer and assert its exact cell count.
	 */
	function getGrayCircles(container: HTMLElement, data: MatrixData): Element[] {
		// compound selector is essential: a page-level ancestor carries the
		// `main` class too, so a bare '.main svg' would also match the sibling
		// .prev layer's svg (it precedes .main in document order)
		const svg = container.querySelector('.main.matrix-container svg.matrix-svg');
		if (!svg) return [];
		const circles = svg.querySelectorAll('circle');
		const cols = data[0]?.length ?? 0;
		if (!cols || circles.length !== data.length * cols) return [];
		const gray: Element[] = [];
		circles.forEach((c, idx) => {
			const i = Math.floor(idx / cols);
			const j = idx % cols;
			if (data[i] && !Number.isFinite(data[i][j])) gray.push(c);
		});
		return gray;
	}

	$: qkColorScale = (d, i) => {
		return d3
			.scaleLinear()
			.domain(qkColorScaleDomain)
			.range(['white', theme.colors['purple'][700]])(d);
	};
	const maskedColorScale = (d, i) => {
		return d3.scaleLinear().domain([-3, 3]).range(['white', theme.colors['purple'][700]])(d);
	};
	const softmaxColorScale = (d, i) => {
		return d3.interpolate('white', theme.colors['purple'][700])(d);
	};

	const onMouseOverCell = (e, d, el) => {
		const rowIdx = d.rowIndex;
		const colIdx = d.colIndex;
		hoveredMatrixCell.set({ row: rowIdx, col: colIdx });
		if (Number.isFinite(d.cell)) {
			d3.select(el).attr('stroke', theme.colors.gray[400]);
		}
	};
	const onMouseOutCell = (e, d, el) => {
		hoveredMatrixCell.set({ row: null, col: null });
		if (Number.isFinite(d.cell)) {
			d3.select(el).attr('stroke', !Number.isFinite(d.cell) ? 'none' : theme.colors.gray[200]);
		}
	};

	const showTooltip = (e, d) => {
		if (!Number.isFinite(d)) return;
		return d.toFixed(2);
	};
</script>

<div
	class="flex items-center gap-8 px-5"
	style={`--attention-matrix-width: ${attentionMatrixWidth}px;`}
	data-click="attention-matrix"
>
	<!-- QK -->
	<div
		role="none"
		class={classNames('attention-matrix-container relative flex', {
			active: isAttentionExpanded
		})}
		bind:this={expandableEl}
		on:click={onClickAttention}
		on:keydown={onClickAttention}
	>
		<div
			class={classNames('attention-matrix attention-qk flex flex-col items-center', {
				'attention-initial': isAttentionExpanded
			})}
			bind:this={attentionQK}
		>
			{#key matrixKey}
				<Matrix
					className="main"
					data={queryKey}
					showSize={false}
					cellHeight={cellSize}
					cellWidth={cellSize}
					rowGap={3}
					colGap={3}
					shape={'circle'}
					colorScale={qkColorScale}
					{onMouseOverCell}
					{onMouseOutCell}
					{showTooltip}
				/>
			{/key}
			<TextbookTooltip id="masked-self-attention">
				<div class="matrix-label">
					{#if $attentionVariant === 'dsa'}Indexer scores{:else if $attentionVariant === 'mla'
					}Up-projected · Dot product{:else}Dot product{/if}
				</div>
			</TextbookTooltip>

			{#if !useOnnx && !vm}
				<!-- variant activations only exist after a real ONNX run
					(cached example data ships MHA tensors only) -->
				<div class="variant-hint">
					Run the model (press Enter) to compute {$attentionVariant.toUpperCase()} attention
				</div>
			{/if}

			<Tooltip class="popover tooltip">
				{#if $attentionVariant === 'gqa'}
					<!-- query head a shares KV head g(a) with its group -->
					<Katex math={'Q_a \\cdot K_{g(a)}^T'}></Katex>
				{:else}
					<Katex math={'Q \\cdot K^T'}></Katex>
				{/if}
			</Tooltip>
			<div class="color-scale">
				<span class="val">{hasQkData ? qkColorScaleDomain[0].toFixed(1) : '—'}</span>
				<div class="bar"></div>
				<span class="val">{hasQkData ? qkColorScaleDomain[1].toFixed(1) : '—'}</span>
			</div>
		</div>
		<!-- Scaling · Mask -->
		<div
			class="attention-matrix attention-mask flex flex-col items-center"
			bind:this={attentionMask}
		>
			<svg
				class="arrow"
				aria-hidden="true"
				xmlns="http://www.w3.org/2000/svg"
				width="24"
				height="24"
				fill="none"
				viewBox="0 0 24 24"
			>
				<path
					stroke="currentColor"
					stroke-linecap="round"
					stroke-linejoin="round"
					stroke-width="2"
					d="M19 12H5m14 0-4 4m4-4-4-4"
				/>
			</svg>

			<div>
				{#key matrixKey}
					<Matrix
						className="prev absolute top-0 left-0 pointer-events-none"
						data={queryKey}
						showSize={false}
						cellHeight={cellSize}
						cellWidth={cellSize}
						rowGap={3}
						colGap={3}
						shape={'circle'}
						colorScale={qkColorScale}
						{onMouseOverCell}
						{onMouseOutCell}
						{showTooltip}
					/>
					<Matrix
						className="main opacity-0"
						data={maskArray(masked)}
						showSize={false}
						cellHeight={cellSize}
						cellWidth={cellSize}
						rowGap={3}
						colGap={3}
						shape={'circle'}
						colorScale={maskedColorScale}
						{onMouseOverCell}
						{onMouseOutCell}
						{showTooltip}
					/>
				{/key}
			</div>
			<TextbookTooltip id="masked-self-attention">
				<div class="matrix-label">
					{#if $attentionVariant === 'dsa'}
						Indexer · Top-k
						<span class="kept-chip">{$dsaTopK} of {$tokens.length} kept / row</span>
					{:else if $attentionVariant === 'swa'}
						Sliding Window
						<span class="kept-chip">{$swaWindowSize} of {$tokens.length} / row</span>
					{:else}Scaling · Mask{/if}
				</div>
			</TextbookTooltip>

			<Tooltip class="popover tooltip">
				{#if $attentionVariant === 'dsa'}
					<!-- keep the top-k indexer scores per row, mask the rest -->
					<Katex math={dsaTopKMath}></Katex>
				{:else if $attentionVariant === 'swa'}
					<!-- each token sees only the last w tokens (incl. itself) -->
					<Katex math={'i - w < j \\le i'}></Katex>
				{:else}
					<Katex math={'\\frac{QK^T}{\\sqrt{d_k}} + M'}></Katex>
				{/if}
			</Tooltip>
			<div class="color-scale">
				<span class="val">-3.0</span>
				<div class="bar"></div>
				<span class="val">3.0</span>
			</div>
		</div>

		<!-- Softmax -->
		<div
			class={classNames('attention-matrix attention-softmax flex flex-col items-center', {
				'attention-out': isAttentionExpanded
			})}
			bind:this={attentionSoftmax}
		>
			<svg
				class="arrow"
				aria-hidden="true"
				xmlns="http://www.w3.org/2000/svg"
				width="24"
				height="24"
				fill="none"
				viewBox="0 0 24 24"
			>
				<path
					stroke="currentColor"
					stroke-linecap="round"
					stroke-linejoin="round"
					stroke-width="2"
					d="M19 12H5m14 0-4 4m4-4-4-4"
				/>
			</svg>
			<div>
				{#key matrixKey}
					<Matrix
						className="prev absolute top-0 left-0  pointer-events-none"
						data={maskArray(masked)}
						showSize={false}
						cellHeight={cellSize}
						cellWidth={cellSize}
						rowGap={3}
						colGap={3}
						shape={'circle'}
						colorScale={maskedColorScale}
						{onMouseOverCell}
						{onMouseOutCell}
						{showTooltip}
					/>
					<Matrix
						className="main opacity-0"
						data={maskArray(softmaxed)}
						showSize={false}
						cellHeight={cellSize}
						cellWidth={cellSize}
						rowGap={3}
						colGap={3}
						shape={'circle'}
						colorScale={softmaxColorScale}
						{onMouseOverCell}
						{onMouseOutCell}
						{showTooltip}
					/>
				{/key}
			</div>

			<TextbookTooltip id="masked-self-attention">
				<div class="matrix-label">Softmax</div>
			</TextbookTooltip>
			<Tooltip class="popover tooltip">
				<Katex math={'\\text{softmax}(\\frac{QK^T}{\\sqrt{d_k}} + M)'}></Katex>
			</Tooltip>
			<div class="color-scale">
				<span class="val">0.0</span>
				<div class="bar"></div>
				<span class="val">1.0</span>
			</div>
		</div>
		<div
			class={classNames('attention-matrix attention-result flex flex-col items-center', {
				'attention-initial': !isAttentionExpanded,
				'attention-out': !isAttentionExpanded
			})}
			bind:this={attentionResult}
			bind:offsetWidth={attentionMatrixWidth}
		>
			{#key matrixKey}
				<Matrix
					className="main"
					data={maskArray(softmaxed)}
					showSize={false}
					cellHeight={cellSize}
					cellWidth={cellSize}
					rowGap={3}
					colGap={3}
					shape={'circle'}
					colorScale={softmaxColorScale}
					{onMouseOverCell}
					{onMouseOutCell}
					{showTooltip}
				/>
			{/key}

			<div class="matrix-label flex items-center gap-1">
				Attention <ZoomInOutline></ZoomInOutline>
			</div>
		</div>
	</div>
</div>

<style lang="scss">
	.attention-matrix-container {
		cursor: pointer;
		border-radius: 0.5rem;
		transition: 0.2s background-color;
		gap: 1rem;
		padding: 1rem;
	}

	.attention-matrix-container {
		.attention-qk,
		.attention-mask,
		.attention-softmax {
			display: none;
		}

		:global(.matrix) {
			padding: 0.5rem;
		}
		.matrix-label {
			white-space: nowrap;
			color: theme('colors.gray.400');

			// DSA: at-a-glance sparsity summary next to "Indexer · Top-k"
			.kept-chip {
				display: inline-block;
				margin-left: 0.35rem;
				padding: 0.05rem 0.4rem;
				border-radius: 999px;
				background: theme('colors.purple.50');
				border: 1px solid theme('colors.purple.200');
				color: theme('colors.purple.700');
				font-size: 0.58rem;
				font-weight: 600;
				vertical-align: 1px;
			}
		}
		.variant-hint {
			font-size: 0.7rem;
			color: theme('colors.amber.600');
			white-space: nowrap;
		}
		.attention-result {
			.matrix-label:hover {
				color: theme('colors.gray.600');
			}
		}
		.arrow {
			position: absolute;
			left: -1rem;
			top: calc(var(--attention-matrix-width) / 2 - 0.5rem);
			width: 1.2rem;
			height: 1.2rem;
			color: theme('colors.gray.300');
		}

		.attention-matrix {
			position: relative;
		}
	}
	.color-scale {
		position: absolute;
		bottom: -1.2rem;
		height: 1rem;
		display: flex;
		justify-content: space-between;
		align-items: center;
		width: 100%;
		padding: 0 1rem;
		gap: 0.2rem;

		.bar {
			height: 0.4rem;
			flex: 1 0 0;
			border: 1px solid theme('colors.gray.200');
			background: linear-gradient(90deg, white 0%, theme('colors.purple.700') 100%);
		}
		.val {
			flex-shrink: 0;
			font-family: monospace;
			font-size: 0.7rem;
			color: theme('colors.gray.600');
		}
	}
</style>
