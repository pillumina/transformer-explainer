<script lang="ts">
	import HeadStack from '~/components/HeadStack.svelte';
	import {
		tokens,
		modelMeta,
		headContentHeight,
		expandedBlock,
		headGap,
		hoveredMatrixCell,
		attentionHeadIdx,
		attentionVariant,
				gqaNumKVHeads
		} from '~/store';
	import classNames from 'classnames';
	import AttentionMatrix from '~/components/AttentionMatrix.svelte';
	import VariantControls from '~/components/VariantControls.svelte';
	import { kvHeadForQuery, GROUP_COLORS } from '~/utils/attentionVariants';

	import { setContext, getContext } from 'svelte';
	import { Tooltip } from 'flowbite-svelte';
	import { onClickReadMore } from '~/utils/event';
	import TextbookTooltip from './common/TextbookTooltip.svelte';

	export let className: string | undefined = undefined;

	setContext('block-id', 'attention');
	const blockId = getContext('block-id');
	$: isAttentionExpanded = $expandedBlock.id === blockId;

	// GQA: the query head h reads from KV head g(h); the K/V vector shown here is
	// that group's slice, split horizontally into n_kv segments (one per KV head).
	$: headNum = $modelMeta.attention_head_num;
	$: nKV = $attentionVariant === 'gqa' ? $gqaNumKVHeads : headNum;
	$: kvHeadIdx = kvHeadForQuery($attentionHeadIdx, headNum, $gqaNumKVHeads);
	$: groupColorOf = (g: number) => GROUP_COLORS[g % GROUP_COLORS.length];
	$: queriesPerKV = headNum / nKV;
	$: groupStart = kvHeadIdx * queriesPerKV;
	$: groupEnd = groupStart + queriesPerKV - 1;
	$: variantTitle = {
		mha: 'Multi-head Self Attention',
		gqa: 'Grouped-Query Attention',
		swa: 'Sliding-Window Attention',
		dsa: 'DeepSeek Sparse Attention',
		mla: 'Multi-head Latent Attention'
	}[$attentionVariant];

	const queryHeadVectorColor = 'bg-blue-400';
	const keyHeadVectorColor = 'bg-red-400';
	const valHeadVectorColor = 'bg-green-400';

	const outputVectorColor = 'bg-purple-500';

	let isHovered = false;

	function handleMouseEnter() {
		isHovered = true;
	}

	function handleMouseLeave() {
		isHovered = false;
	}
</script>

<div
	class={classNames('attention', className, {
		expanded: isAttentionExpanded
	})}
	data-click="attention-step"
>
	<div
		class="title"
		on:mouseenter={handleMouseEnter}
		on:mouseleave={handleMouseLeave}
		role="group"
		data-click="attention-step-title"
	>
		<div class="title-row">
			<div class="w-max">
				<TextbookTooltip id="self-attention">{variantTitle}</TextbookTooltip>
			</div>
			<VariantControls className="variant-controls" />
		</div>
	</div>
	<div class="content relative">
		<div
			class="bounding attention-bounding"
			class:active={isHovered && !isAttentionExpanded}
			style={`padding-bottom:${$modelMeta.attention_head_num * headGap.y}px`}
		></div>
		<div class="heads">
			<HeadStack>
				<div
					class="head-block relative flex w-full items-center justify-between px-2"
					style={`height:${$headContentHeight}px;`}
				>
					<!-- GQA sharing note lives in HeadGrid (active-group caption),
						so it can never collide with the head-block layout -->
					<div class="qkv flex h-full flex-col justify-center gap-[5rem] pl-[6rem]">
						<div class="column key">
							<div class="head1 title">
								<TextbookTooltip id="qkv"
									>{#if $attentionVariant === 'gqa'}Key · KV {kvHeadIdx + 1} of {nKV}{:else}Key{/if}</TextbookTooltip
								>
							</div>

							{#each $tokens as token, index}
								<div
									class="head1 key cell x1-12 text-xs"
									class:last={index === $tokens.length - 1}
									class:active={$hoveredMatrixCell.col === index}
								>
									<span class="label float">{token}</span>
									{#if $attentionVariant === 'gqa'}
										<!-- one K vector per token, split across the n_kv KV heads -->
										<div class="vector gqa-strip">
											{#each Array(nKV) as _, g}
												<div
													class="gqa-seg"
													class:active={g === kvHeadIdx}
													style={`width:${100 / nKV}%;background:${groupColorOf(g)};`}
												></div>
											{/each}
										</div>
									{:else}
										<div class={`vector x1-12 ${keyHeadVectorColor}`}></div>
									{/if}
								</div>
							{/each}
							<Tooltip class="popover" triggeredBy={'.step.attention .key .cell'} placement="right"
								>{#if $attentionVariant === 'gqa'}Key, KV Head {kvHeadIdx + 1} of {nKV}; dark
									segment = this query head's KV group, light = other KV heads{:else}Key, Head {$attentionHeadIdx +
										1}, vector({$modelMeta.dimension / $modelMeta.attention_head_num}){/if}</Tooltip
							>
						</div>
						<div class="column query">
							<div class="head1 title"><TextbookTooltip id="qkv">Query</TextbookTooltip></div>
							{#each $tokens as token, index}
								<div
									class="head1 cell x1-12 query text-xs"
									class:last={index === $tokens.length - 1}
									class:active={$hoveredMatrixCell.row === index}
								>
									<span class="label float">{token}</span>
									<div class={`vector x1-12  ${queryHeadVectorColor}`}></div>
								</div>
							{/each}
							<Tooltip
								class="popover"
								triggeredBy={'.step.attention .query .cell'}
								placement="right"
								>Query, Head {$attentionHeadIdx + 1}, vector({$modelMeta.dimension /
									$modelMeta.attention_head_num})</Tooltip
							>
						</div>
						<div class="column value">
							<div class="head1 title">
								<TextbookTooltip id="qkv"
									>{#if $attentionVariant === 'gqa'}Value · KV {kvHeadIdx + 1} of {nKV}{:else}Value{/if}</TextbookTooltip
								>
							</div>
							{#each $tokens as token, index}
								<div class="head1 cell x1-12 text-xs" class:last={index === $tokens.length - 1}>
									<span class="label float">{token}</span>
									{#if $attentionVariant === 'gqa'}
										<!-- one V vector per token, split across the n_kv KV heads -->
										<div class="vector gqa-strip">
											{#each Array(nKV) as _, g}
												<div
													class="gqa-seg"
													class:active={g === kvHeadIdx}
													style={`width:${100 / nKV}%;background:${groupColorOf(g)};`}
												></div>
											{/each}
										</div>
									{:else}
										<div class={`vector x1-12 ${valHeadVectorColor}`}></div>
									{/if}
								</div>
							{/each}
							<Tooltip
								class="popover"
								triggeredBy={'.step.attention .value .cell'}
								placement="right"
								>{#if $attentionVariant === 'gqa'}Value, KV Head {kvHeadIdx + 1} of {nKV}; dark
									segment = this query head's KV group{:else}Value, Head {$attentionHeadIdx + 1},
									vector({$modelMeta.dimension / $modelMeta.attention_head_num}){/if}</Tooltip
							>
						</div>
					</div>
					<div class="resize-watch attention-matrix flex">
						<AttentionMatrix />
					</div>
					<div class="head-out mx-[2rem]">
						<div class="column out">
							<div class="head1 title">
								<TextbookTooltip id="output-concatenation">Out</TextbookTooltip>
							</div>
							{#each $tokens as token, index}
								<div class="head1 cell x1-12" class:last={index === $tokens.length - 1}>
									<div class={`vector x1-12 ${outputVectorColor}`}></div>
								</div>
							{/each}
							<Tooltip class="popover" triggeredBy={'.step.attention .out .cell'} placement="right"
								>Attention Out, Head 1, vector({$modelMeta.dimension /
									$modelMeta.attention_head_num})</Tooltip
							>
						</div>
					</div>
				</div>
			</HeadStack>
		</div>
	</div>
</div>

<style lang="scss">
	.attention-matrix,
	.head-title {
		z-index: $COLUMN_TITLE_INDEX;
	}
	.attention {
		> .title > div {
			// cursor: help;
		}
		.title-row {
			display: flex;
			align-items: center;
			justify-content: center;
			gap: 0.75rem;
			flex-wrap: wrap;
		}
		.column .title {
			white-space: nowrap;
		}
		// GQA: one K/V vector per token, split horizontally across n_kv KV heads.
		// The active KV head (this query head's group) is solid; the rest are
		// light so the sharing structure reads at a glance.
		// Width/height mirror the global .vector token (12px × height/12)
		// so the Sankey line anchors stay aligned with MHA mode.
		.gqa-strip {
			height: calc(var(--vector-height) / 12);
			width: 12px;
			display: flex;
			overflow: hidden;
			border-radius: 0.1rem;
			flex-shrink: 0;
		}
		.gqa-seg {
			height: 100%;
			opacity: 0.25;
			transition: opacity 0.2s;

			&.active {
				opacity: 1;
			}
		}
		// GQA sharing note moved into HeadGrid's active-group caption — kept
		// out of this component so it can never collide with head-block layout
		.kv-share-dot {
			width: 0.55rem;
			height: 0.55rem;
			border-radius: 999px;
			flex-shrink: 0;
		}
		&.expanded {
			.title,
			:global(.head-content) {
				z-index: $EXPANDED_CONTENT_INDEX;
			}
			:global(.multi-head .head-card:first-child) {
				z-index: $EXPANDED_CONTENT_INDEX !important;
			}
		}

		.attention-bounding {
			top: -0.5rem;
			padding: 0.5rem 0;
			left: -0.3rem;
			width: calc(100% + 1rem);
			height: calc(100%);
		}
		.column {
			.label {
				font-size: 0.7rem;
				color: theme('colors.gray.600');
			}
			.title {
				z-index: $COLUMN_TITLE_INDEX;
				position: absolute;
				top: -1.7rem;
				left: 50%;
				transform: translateX(-50%);
				font-size: 0.9rem;
				transition: none;
			}
			&.query .title {
				color: theme('colors.blue.400');
			}
			&.key .title {
				color: theme('colors.red.400');
			}
			&.value .title {
				color: theme('colors.green.400');
			}
			&.out .title {
				color: theme('colors.purple.400');
			}
		}
		.content {
			display: grid;
			grid-template-columns: auto 0;

			.tokens {
				gap: 0.6rem;
			}
		}
		.heads {
			padding: 0 7rem 0 8rem;

			.head1.cell {
				.label {
					height: auto;
					line-height: 1;
				}
				&.active {
					&.query {
						.label {
							background-color: theme('colors.blue.100');
							color: theme('colors.blue.700');
							font-size: 1rem;
							z-index: 100;
							padding: 0.2rem;
						}
					}
					&.key {
						.label {
							background-color: theme('colors.red.100');
							color: theme('colors.red.700');
							font-size: 1rem;
							z-index: 100;
							padding: 0.2rem;
						}
					}
				}
			}
		}
	}
</style>
