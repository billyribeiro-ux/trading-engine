<script lang="ts">
	import type { Dissection } from "$lib/types";
	import {
		CrosshairIcon,
		PulseIcon,
		ScalesIcon,
		StackIcon,
		TargetIcon,
		TrendDownIcon,
		TrendUpIcon,
		WarningCircleIcon
	} from "phosphor-svelte";

	let { dissection }: { dissection: Dissection } = $props();
	const h = $derived(dissection.header);
</script>

<section class="dissection">
	<header class="head">
		<div class="title">
			<h2><CrosshairIcon weight="duotone" size="1em" /> {dissection.symbol}</h2>
			<span class="date">{dissection.date}</span>
			<span class="bias">{h.bias}</span>
		</div>
		<div class="ohlc">
			<span>O <b>{h.open}</b></span>
			<span>H <b>{h.high}</b></span>
			<span>L <b>{h.low}</b></span>
			<span>C <b>{h.close}</b></span>
			<span>Range <b>{h.range} ({h.range_pct}%, {h.range_atr} ATR)</b></span>
			<span>VWAP <b>{h.vwap_close}</b></span>
		</div>
		{#if !dissection.consistent}
			<p class="warn">
				<WarningCircleIcon size={15} weight="fill" />
				STRUCTURE and LEG ROLES disagree — inconsistent dissection.
			</p>
		{/if}
	</header>

	{#if dissection.read}
		<p class="read">{dissection.read}</p>
	{/if}

	<div class="grid">
		<div class="panel">
			<h3><StackIcon size={15} /> Structure <small>{dissection.structure.length} legs</small></h3>
			<ol class="legs">
				{#each dissection.structure as leg (leg.n)}
					<li>
						<span class="arrow {leg.direction}">
							{#if leg.direction === "up"}
								<TrendUpIcon size={13} weight="bold" />
							{:else}
								<TrendDownIcon size={13} weight="bold" />
							{/if}
						</span>
						<span class="time">{leg.start_time}→{leg.end_time}</span>
						<span class="px">{leg.start_price} → {leg.end_price}</span>
						<span class="mag">{leg.magnitude} pts</span>
						{#if leg.sub_legs.length}
							<span class="sub">+{leg.sub_legs.length} internal</span>
						{/if}
					</li>
				{/each}
			</ol>
		</div>

		<div class="panel">
			<h3><ScalesIcon size={15} /> Leg roles</h3>
			<ol class="legs">
				{#each dissection.leg_roles as role (role.n)}
					<li>
						<span class="arrow {role.direction}">
							{#if role.direction === "up"}
								<TrendUpIcon size={13} weight="bold" />
							{:else}
								<TrendDownIcon size={13} weight="bold" />
							{/if}
						</span>
						<span class="atr">{role.magnitude_atr} ATR</span>
						<span class="vwap">VWAP {role.vwap_start}→{role.vwap_end}</span>
						<span class="roles">{role.roles.join(", ")}</span>
					</li>
				{/each}
			</ol>
		</div>

		<div class="panel">
			<h3><PulseIcon size={15} /> VWAP map</h3>
			{#if dissection.vwap_map.length}
				<ul class="events">
					{#each dissection.vwap_map as e, i (i)}
						<li>
							<span class="time">{e.first_time}{e.count > 1 ? `–${e.last_time} ×${e.count}` : ""}</span>
							<span class="etype">{e.type}</span>
							<span class="oc">{e.outcome_atr > 0 ? "+" : ""}{e.outcome_atr} ATR</span>
						</li>
					{/each}
				</ul>
			{:else}
				<p class="muted">No VWAP interactions.</p>
			{/if}
		</div>

		<div class="panel">
			<h3><TargetIcon size={15} /> Key levels</h3>
			{#if dissection.levels.length}
				<ul class="events">
					{#each dissection.levels as lv, i (i)}
						<li>
							<span class="time">{lv.time}</span>
							<span class="kind">{lv.kind.replaceAll("_", " ")}</span>
							<span class="lvl">@ {lv.level}</span>
							<span class="verdict">{lv.held ? "HELD" : "BROKE"}</span>
						</li>
					{/each}
				</ul>
			{:else}
				<p class="muted">No level tests.</p>
			{/if}
		</div>
	</div>
</section>

<style>
	.dissection {
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}
	.head .title {
		display: flex;
		align-items: baseline;
		gap: 0.75rem;
	}
	.head h2 {
		margin: 0;
		font-size: 1.4rem;
		display: flex;
		align-items: center;
		gap: 0.4rem;
	}
	.head h2 :global(svg) {
		color: var(--accent);
	}
	.date {
		color: var(--muted);
	}
	.bias {
		font-size: 0.75rem;
		padding: 0.1rem 0.5rem;
		border: 1px solid var(--border);
		border-radius: 999px;
	}
	.ohlc {
		display: flex;
		flex-wrap: wrap;
		gap: 1rem;
		margin-top: 0.4rem;
		color: var(--muted);
		font-size: 0.9rem;
	}
	.ohlc b {
		color: var(--fg);
	}
	.read {
		font-style: italic;
		color: var(--fg-soft);
		border-left: 2px solid var(--accent);
		padding-left: 0.75rem;
		margin: 0;
	}
	.warn {
		color: var(--down);
		font-size: 0.85rem;
		display: flex;
		align-items: center;
		gap: 0.3rem;
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
		gap: 1rem;
	}
	.panel {
		background: var(--bg-panel);
		border: 1px solid var(--border);
		border-radius: 8px;
		padding: 0.75rem 1rem;
	}
	.panel h3 {
		margin: 0 0 0.5rem;
		font-size: 0.95rem;
		display: flex;
		align-items: center;
		gap: 0.4rem;
	}
	.panel h3 small {
		color: var(--muted);
		font-weight: normal;
	}
	.legs,
	.events {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 0.3rem;
		font-size: 0.85rem;
		font-variant-numeric: tabular-nums;
	}
	.legs li,
	.events li {
		display: flex;
		gap: 0.6rem;
		align-items: baseline;
	}
	.arrow {
		display: inline-flex;
		align-items: center;
	}
	.arrow.up {
		color: var(--up);
	}
	.arrow.down {
		color: var(--down);
	}
	.time {
		color: var(--muted);
		min-width: 5.5rem;
	}
	.roles {
		color: var(--accent);
	}
	.muted {
		color: var(--muted);
		font-size: 0.85rem;
	}
</style>
