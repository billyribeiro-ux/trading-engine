<script lang="ts">
	import { dissect, screen } from "$lib/api";
	import DissectionView from "$lib/components/DissectionView.svelte";
	import JournalView from "$lib/components/JournalView.svelte";
	import type { Dissection, ScreenResponse } from "$lib/types";
	import {
		CalendarBlankIcon,
		ChartBarIcon,
		ChartLineUpIcon,
		CrosshairIcon,
		FlaskIcon,
		FunnelIcon,
		GaugeIcon,
		ListChecksIcon,
		MagnifyingGlassIcon,
		SlidersHorizontalIcon,
		StackIcon,
		TimerIcon,
		TrendDownIcon,
		TrendUpIcon
	} from "phosphor-svelte";

	const LOOKBACK_DEFAULT: Record<string, number> = { intraday: 60, swing: 730, portfolio: 1825 };

	let scanner = $state("intraday");
	let watchlist = $state("TSLA, NVDA, AAPL");
	let timeframe = $state("5min");
	let style = $state("reversal");
	let lookbackDays = $state(60);
	let probaThreshold = $state(0.55);
	let fdr = $state(0.1);

	function onScannerChange() {
		lookbackDays = LOOKBACK_DEFAULT[scanner] ?? 60;
	}

	let screening = $state(false);
	let screenError = $state<string | null>(null);
	let result = $state<ScreenResponse | null>(null);

	let selected = $state<string | null>(null);
	let dissecting = $state(false);
	let dissectError = $state<string | null>(null);
	let dissection = $state<Dissection | null>(null);

	const symbols = $derived(
		watchlist
			.split(/[\s,]+/)
			.map((s) => s.trim().toUpperCase())
			.filter(Boolean)
	);

	async function runScreen() {
		if (!symbols.length) return;
		screening = true;
		screenError = null;
		result = null;
		try {
			result = await screen({
				symbols,
				scanner,
				timeframe,
				style,
				lookback_days: lookbackDays,
				proba_threshold: probaThreshold,
				fdr
			});
		} catch (e) {
			screenError = e instanceof Error ? e.message : String(e);
		} finally {
			screening = false;
		}
	}

	async function openDissection(symbol: string) {
		selected = symbol;
		dissecting = true;
		dissectError = null;
		dissection = null;
		try {
			dissection = await dissect(symbol, timeframe);
		} catch (e) {
			dissectError = e instanceof Error ? e.message : String(e);
		} finally {
			dissecting = false;
		}
	}

	const pct = (x: number) => `${(x * 100).toFixed(0)}%`;
	const sign = (x: number) => `${x >= 0 ? "+" : ""}${x.toFixed(3)}`;
</script>

<svelte:head><title>Trading Engine — Scanner</title></svelte:head>

<main>
	<h1>
		<ChartLineUpIcon weight="duotone" size="1.4em" />
		Trading Engine <span class="sub">scanner</span>
	</h1>

	<form
		class="controls"
		onsubmit={(e) => {
			e.preventDefault();
			runScreen();
		}}
	>
		<label>
			<span class="lbl"><FunnelIcon size={13} /> Scanner</span>
			<select bind:value={scanner} onchange={onScannerChange}>
				<option value="intraday">intraday</option>
				<option value="swing">swing</option>
				<option value="portfolio">portfolio</option>
			</select>
		</label>
		<label class="grow">
			<span class="lbl"><StackIcon size={13} /> Watchlist</span>
			<input bind:value={watchlist} placeholder="TSLA, NVDA, AAPL" />
		</label>
		<label>
			<span class="lbl"><CalendarBlankIcon size={13} /> Lookback (days)</span>
			<input type="number" min="5" max="3650" bind:value={lookbackDays} />
		</label>
		{#if scanner === "intraday"}
			<label>
				<span class="lbl"><TimerIcon size={13} /> Timeframe</span>
				<select bind:value={timeframe}>
					<option>1min</option>
					<option>5min</option>
					<option>15min</option>
					<option>30min</option>
					<option>1hour</option>
				</select>
			</label>
			<label>
				<span class="lbl"><SlidersHorizontalIcon size={13} /> Style</span>
				<select bind:value={style}>
					<option value="reversal">reversal</option>
					<option value="scalp">scalp</option>
				</select>
			</label>
		{/if}
		<label>
			<span class="lbl"><GaugeIcon size={13} /> Min prob</span>
			<input type="number" min="0.5" max="0.99" step="0.01" bind:value={probaThreshold} />
		</label>
		<label>
			<span class="lbl"><FlaskIcon size={13} /> FDR</span>
			<input type="number" min="0.01" max="0.5" step="0.01" bind:value={fdr} />
		</label>
		<button type="submit" disabled={screening || !symbols.length}>
			<MagnifyingGlassIcon size={16} weight="bold" />
			{screening ? "Screening…" : "Run screen"}
		</button>
	</form>

	{#if screenError}
		<p class="error">Screen failed: {screenError}</p>
	{/if}

	{#if result}
		<p class="summary">
			<ListChecksIcon size={15} />
			{result.summary.configs_evaluated} configs evaluated · {result.summary.survived} survived ·
			<b>{result.summary.n_signals} signals</b>
		</p>

		{#if result.signals.length}
			<table class="signals">
				<thead>
					<tr>
						<th>Symbol</th>
						<th>Dir</th>
						<th>Event</th>
						<th class="num">Entry</th>
						<th class="num">Stop</th>
						<th class="num">Target</th>
						<th class="num">R:R</th>
						<th class="num">Prob</th>
						<th class="num">Edge R</th>
						<th class="num">p(fdr)</th>
						<th class="num">Decay</th>
					</tr>
				</thead>
				<tbody>
					{#each result.signals as s, i (i)}
						<tr class:active={selected === s.symbol}>
							<td>
								<button class="link" onclick={() => openDissection(s.symbol)}>
									<CrosshairIcon size={13} />{s.symbol}
								</button>
							</td>
							<td>
								<span class="dir {s.direction}">
									{#if s.direction === "long"}
										<TrendUpIcon size={12} weight="bold" />
									{:else}
										<TrendDownIcon size={12} weight="bold" />
									{/if}
									{s.direction}
								</span>
							</td>
							<td class="event">{s.event_type}</td>
							<td class="num">{s.entry.toFixed(2)}</td>
							<td class="num">{s.stop.toFixed(2)}</td>
							<td class="num">{s.target.toFixed(2)}</td>
							<td class="num">{s.rr.toFixed(2)}</td>
							<td class="num">{pct(s.probability)}</td>
							<td class="num pos">{sign(s.oos_edge_r)}</td>
							<td class="num">{s.p_value_fdr.toFixed(3)}</td>
							<td class="num">{sign(s.decay)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		{:else}
			<p class="muted">
				No validated signals — no significant edge survived the screen. That is a correct, honest
				result.
			</p>
		{/if}

		<details class="reports">
			<summary><ChartBarIcon size={14} /> All evaluated configs ({result.reports.length})</summary>
			<table>
				<thead>
					<tr>
						<th>Symbol</th>
						<th class="num">Events</th>
						<th class="num">Signals</th>
						<th class="num">Edge R</th>
						<th class="num">AUC</th>
						<th class="num">p</th>
						<th class="num">p(fdr)</th>
					</tr>
				</thead>
				<tbody>
					{#each result.reports as r, i (i)}
						<tr>
							<td>{r.symbol}</td>
							<td class="num">{r.n_events}</td>
							<td class="num">{r.n_signals}</td>
							<td class="num">{sign(r.oos_edge_r)}</td>
							<td class="num">{r.oos_auc.toFixed(3)}</td>
							<td class="num">{r.p_value.toFixed(3)}</td>
							<td class="num">{r.p_value_fdr.toFixed(3)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</details>
	{/if}

	{#if selected}
		<hr />
		<div class="dissect-wrap">
			{#if dissecting}
				<p class="muted">Dissecting {selected}…</p>
			{:else if dissectError}
				<p class="error">Dissection failed: {dissectError}</p>
			{:else if dissection}
				<DissectionView {dissection} />
			{/if}
		</div>
	{/if}

	<hr />
	<JournalView />
</main>

<style>
	main {
		max-width: 1100px;
		margin: 0 auto;
		padding: 1.5rem;
		display: flex;
		flex-direction: column;
		gap: 1.25rem;
	}
	h1 {
		margin: 0;
		font-size: 1.6rem;
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}
	h1 :global(svg) {
		color: var(--accent);
	}
	h1 .sub {
		color: var(--muted);
		font-weight: 300;
	}
	.lbl {
		display: inline-flex;
		align-items: center;
		gap: 0.3rem;
	}
	.controls {
		display: flex;
		flex-wrap: wrap;
		gap: 0.75rem;
		align-items: end;
		background: var(--bg-panel);
		border: 1px solid var(--border);
		border-radius: 8px;
		padding: 1rem;
	}
	label {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		font-size: 0.8rem;
		color: var(--muted);
	}
	label.grow {
		flex: 1;
		min-width: 220px;
	}
	input,
	select {
		background: var(--bg);
		color: var(--fg);
		border: 1px solid var(--border);
		border-radius: 6px;
		padding: 0.4rem 0.5rem;
		font: inherit;
	}
	button[type="submit"] {
		background: var(--accent);
		color: #04111f;
		border: none;
		border-radius: 6px;
		padding: 0.55rem 1.1rem;
		font-weight: 600;
		cursor: pointer;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		gap: 0.4rem;
	}
	button[type="submit"]:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	.summary {
		color: var(--muted);
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 0.4rem;
	}
	.summary b {
		color: var(--fg);
	}
	table {
		width: 100%;
		border-collapse: collapse;
		font-size: 0.85rem;
		font-variant-numeric: tabular-nums;
	}
	th,
	td {
		text-align: left;
		padding: 0.4rem 0.6rem;
		border-bottom: 1px solid var(--border);
	}
	th.num,
	td.num {
		text-align: right;
	}
	tr.active {
		background: color-mix(in srgb, var(--accent) 12%, transparent);
	}
	.link {
		background: none;
		border: none;
		color: var(--accent);
		cursor: pointer;
		font: inherit;
		padding: 0;
		display: inline-flex;
		align-items: center;
		gap: 0.3rem;
	}
	.dir {
		text-transform: uppercase;
		font-size: 0.72rem;
		padding: 0.05rem 0.4rem;
		border-radius: 4px;
		display: inline-flex;
		align-items: center;
		gap: 0.25rem;
	}
	.dir.long {
		color: var(--up);
		background: color-mix(in srgb, var(--up) 15%, transparent);
	}
	.dir.short {
		color: var(--down);
		background: color-mix(in srgb, var(--down) 15%, transparent);
	}
	.pos {
		color: var(--up);
	}
	.muted {
		color: var(--muted);
	}
	.error {
		color: var(--down);
	}
	.reports summary {
		cursor: pointer;
		color: var(--muted);
		display: flex;
		align-items: center;
		gap: 0.4rem;
	}
	hr {
		border: none;
		border-top: 1px solid var(--border);
	}
</style>
