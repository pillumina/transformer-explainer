<script lang="ts">
	import {
		tokens,
		modelMeta,
		attentionHeadIdx,
		vectorHeight,
		blockIdx,
		attentionVariant,
		gqaNumKVHeads,
		mlaLatentDim,
		expandedBlock,
		hoveredMatrixCell
	} from '~/store';
	import classNames from 'classnames';
	import VectorCanvas from './common/VectorCanvas.svelte';
	import OperationGroup from './OperationGroup.svelte';
	import { Tooltip } from 'flowbite-svelte';
	import { tick } from 'svelte';
	import { GROUP_COLORS, kvHeadForQuery } from '~/utils/attentionVariants';

	export let className: string | undefined = undefined;

	const embeddingVectorColor = 'bg-gray-300';

	let vectorHoverIdx: number | null = null;
	const queryVectorColor = 'bg-blue-300';
	const keyVectorColor = 'bg-red-300';
	const valVectorColor = 'bg-green-300';

	const queryHeadVectorColor = 'bg-blue-500';
	const keyHeadVectorColor = 'bg-red-500';
	const valHeadVectorColor = 'bg-green-500';

	$: headNum = $modelMeta.attention_head_num;
	$: isGqa = $attentionVariant === 'gqa';
	$: isMla = $attentionVariant === 'mla';
	$: nKV = isGqa ? $gqaNumKVHeads : headNum;
	$: headSliceH = $vectorHeight / headNum;
	$: kvSliceH = $vectorHeight / nKV;
	$: kvIdx = kvHeadForQuery($attentionHeadIdx, headNum, $gqaNumKVHeads);
	$: groupColor = GROUP_COLORS[kvIdx % GROUP_COLORS.length];
	$: groupOf = (g: number) => GROUP_COLORS[g % GROUP_COLORS.length];

	// query-head cursor: position within the Q strip; tinted by KV group in GQA
	$: qCursorStyle = `top:${$attentionHeadIdx * headSliceH}px;${
		isGqa ? `background:${groupColor};` : ''
	}`;
	// K/V cursor: marks the active KV head (a whole group in GQA mode)
	$: kvCursorIdx = isGqa ? kvIdx : $attentionHeadIdx;
	$: kvCursorStyle = `top:${kvCursorIdx * kvSliceH}px;height:${kvSliceH}px;${
		isGqa ? `background:${groupColor};` : ''
	}`;

	// MLA: when an expansion starts, pulse the violet 'up-projected' stripes on
	// K / V so the "grows out of c_KV" story fires as the matrices appear
	let projectedPulse = false;
	let pulseTimer: ReturnType<typeof setTimeout>;
	$: if ($expandedBlock.id !== null && $attentionVariant === 'mla') triggerProjectedPulse();
	async function triggerProjectedPulse() {
		projectedPulse = false;
		await tick();
		projectedPulse = true;
		clearTimeout(pulseTimer);
		pulseTimer = setTimeout(() => (projectedPulse = false), 2600);
	}

	// cross-highlight: hovering an attention cell outlines the token bar of the
	// QUERY (row, blue) and of the KEY / VALUE (column, amber)
	$: hlQueryRow = $hoveredMatrixCell.row;
	$: hlKeyCol = $hoveredMatrixCell.col;
</script>

<div class={classNames('qkv', className)} role="none" data-click="qkv-step">
	<div class="content relative">
		<div
			class="vector-column block-start-column relative flex"
			class:initial-column={$blockIdx === 0}
		>
			<div class="column vectors embedding-column">
				{#each $tokens as token, index}
					<div
						class={`vector ${$blockIdx !== 0 ? 'bg-blue-200' : embeddingVectorColor}`}
						class:last={index === $tokens.length - 1}
					>
						<VectorCanvas colorScale={$blockIdx !== 0 ? 'blue' : 'gray'} />
					</div>
				{/each}
			</div>
			<Tooltip class="popover" triggeredBy={'.qkv .embedding-column .vector'} placement="right"
				>vector({$modelMeta.dimension})</Tooltip
			>

			<div class="operations flex">
				<OperationGroup type="dropout" id={'embedding-dropout'} />
				<OperationGroup type="residual-start" id={'embedding-residual'} />
				<OperationGroup type="ln" id={'embedding-ln'} />
			</div>
		</div>
		<div class="column qkv-column">
			{#each $tokens as token, index}
				<div
					class="qkv-weighted vector x3 flex flex-col"
					class:last={index === $tokens.length - 1}
					class:hl-q={hlQueryRow === index}
					class:hl-k={hlKeyCol === index}
					on:mouseenter={() => {
						vectorHoverIdx = index;
					}}
					on:mouseleave={() => {
						vectorHoverIdx = null;
					}}
					role="group"
				>
					<div class={`sub-vector query relative flex grow flex-col ${queryVectorColor}`}>
						<VectorCanvas colorScale="blue" active={vectorHoverIdx === index} />
						{#if isGqa}
							{#each Array(nKV - 1) as _, g}
								<div
									class="group-sep"
									style={`top:${(g + 1) * headSliceH * (headNum / nKV)}px`}
								></div>
							{/each}
						{/if}
						<div
							class={`sub-vector head1 ${queryHeadVectorColor} absolute`}
							style={qCursorStyle}
						></div>
						<div class="sub-vector head-rest">
							{#if vectorHoverIdx !== index}<span>Q</span>{/if}
						</div>
					</div>
					{#if isMla}
						<!-- shared compressed latent: the ONLY vector cached per token.
							K and V below are up-projected from it (violet stripe), which
							is why the attention matrices stay identical to MHA. -->
						<div class="sub-vector c-kv"></div>
					{/if}
					<div
						class={`sub-vector key relative flex grow flex-col ${keyVectorColor}`}
						class:projected={isMla}
						class:pulse={projectedPulse}
					>
							<VectorCanvas colorScale="red" active={vectorHoverIdx === index} />
							{#if isGqa}
								{#each Array(nKV) as _, g}
									<div
										class="kv-group absolute"
										class:active={g === kvIdx}
										style={`top:${g * kvSliceH}px;height:${kvSliceH}px;background:${groupOf(g)}`}
									></div>
								{/each}
							{/if}
							<div
								class={`sub-vector head1 ${keyHeadVectorColor} absolute ${isGqa ? 'hide' : ''}`}
								style={kvCursorStyle}
							></div>
							<div class="sub-vector head-rest">
								{#if vectorHoverIdx !== index}<span>K</span>{/if}
							</div>
						</div>
						<div
							class={`sub-vector value relative flex grow flex-col ${valVectorColor}`}
							class:projected={isMla}
							class:pulse={projectedPulse}
						>
							<VectorCanvas colorScale="green" active={vectorHoverIdx === index} />
							{#if isGqa}
								{#each Array(nKV) as _, g}
									<div
										class="kv-group absolute"
										class:active={g === kvIdx}
										style={`top:${g * kvSliceH}px;height:${kvSliceH}px;background:${groupOf(g)}`}
									></div>
								{/each}
							{/if}
							<div
								class={`sub-vector head1 ${valHeadVectorColor} absolute ${isGqa ? 'hide' : ''}`}
								style={kvCursorStyle}
							></div>
							<div class="sub-vector head-rest">
								{#if vectorHoverIdx !== index}<span>V</span>{/if}
							</div>
						</div>
					</div>
			{/each}
			<Tooltip class="popover" triggeredBy={'.qkv .qkv-column .vector'} placement="right"
				>{#if isMla}vector(Q {$modelMeta.dimension} + latent {$mlaLatentDim}) — K / V are
					up-projected from the shared compressed latent c_KV; only c_KV (+ a RoPE key) is cached{:else if isGqa}vector({$modelMeta.dimension *
						3}) — K / V hold only {nKV} KV heads, shared by query head groups{:else}vector({$modelMeta.dimension *
						3}){/if}</Tooltip
			>
			<Tooltip class="popover" triggeredBy={'.qkv .c-kv'} placement="right"
				><b>c_KV</b> — the compressed latent vector ({$mlaLatentDim}d). It is the
				<em>only</em> vector cached per token; K and V (violet stripe) are up-projected from
				it, so the attention matrices stay identical to MHA.</Tooltip
			>
		</div>
	</div>
</div>

<style lang="scss">
	.qkv {
		.content {
			display: grid;
			grid-template-columns: 1fr 1fr;

			.vector-column {
				position: relative;
				left: 3rem;

				&.initial-column {
					left: -12px;
				}
			}
			.qkv-column {
				position: relative;
				left: 12px;
				display: flex;
				flex-direction: column;
				align-items: end;

				&.hide {
					pointer-events: none;
					opacity: 0.2;
				}
				// cross-highlight from attention-matrix cell hover:
				// row = query token (blue), col = key/value token (amber)
				.vector.x3 {
					transition:
						opacity 0.15s,
						outline-color 0.15s;
					outline: 2px solid transparent;
					outline-offset: 1px;

					&.hl-q {
						outline-color: theme('colors.blue.500');
					}
					&.hl-k {
						outline-color: theme('colors.amber.500');
					}
					&.hl-q.hl-k {
						outline-color: theme('colors.purple.500');
					}
				}
				.sub-vector {
					user-select: none;
					font-size: 1rem;
					opacity: 0.8;

					&.c-kv {
						// compressed shared latent: deliberately narrower than Q/K/V,
						// solid purple, no value texture (it is a learned summary).
						// Details via hover tooltip — the bar itself stays clean.
						flex: 0 0 auto;
						height: calc(var(--vector-height) * 0.42);
						background: theme('colors.violet.500');
						border-radius: 0.15rem;
						margin: 3px 0;
						opacity: 1;
						z-index: 3;
					}

					&.projected {
						// violet link to c_KV: K / V are up-projected from the latent
						&::before {
							content: '';
							position: absolute;
							left: 0;
							top: 0;
							bottom: 0;
							width: 3px;
							background: theme('colors.violet.500');
							z-index: 2;
						}

						// fires when the attention expansion starts: the stripes flare
						// once, tying the "up-projected from c_KV" story to the moment
						// the matrices grow out of the compressed latent
						&.pulse::before {
							animation: projected-pulse 1.2s ease-out 0.8s both;
						}
					}
					span {
						opacity: 0.45;
					}
					&.query {
						color: theme('colors.blue.600');
					}
					&.key {
						color: theme('colors.red.600');
					}
					&.value {
						color: theme('colors.green.600');
					}

					.head-rest {
						height: 100%;
						display: flex;
						justify-content: center;
						align-items: center;
						font-weight: 700;
						text-shadow:
							-1px -1px 0 white,
							1px -1px 0 white,
							-1px 1px 0 white,
							1px 1px 0 white;
					}

					.group-sep {
						position: absolute;
						left: 0;
						right: 0;
						height: 2px;
						background: white;
						z-index: 2;
					}

					.kv-group {
						left: 0;
						right: 0;
						opacity: 0.18;
						z-index: 1;

						&.active {
							opacity: 0.85;
							z-index: 2;
						}
					}

					.head1.hide {
						opacity: 0;
					}
				}
			}
		}

		&.animate-forward {
			.vector-column {
				transition-delay: 900ms;
				transition: left 100ms;
			}
		}
	}

	@keyframes projected-pulse {
		0% {
			width: 3px;
			box-shadow: 0 0 0 0 rgba(139, 92, 246, 0);
		}
		40% {
			width: 6px;
			box-shadow: 2px 0 8px 0 rgba(139, 92, 246, 0.85);
		}
		100% {
			width: 3px;
			box-shadow: 0 0 0 0 rgba(139, 92, 246, 0);
		}
	}
</style>
