<script lang="ts">
	import { onMount } from "svelte";
	import { downloadExport, journal, resolveJournal, scanJournal } from "$lib/api";
	import type { JournalResponse } from "$lib/types";
	import {
		ArrowsClockwiseIcon,
		CheckCircleIcon,
		ClockIcon,
		CurrencyDollarIcon,
		FileCsvIcon,
		FileXlsIcon,
		LightningIcon,
		NotebookIcon,
		TrendDownIcon,
		TrendUpIcon
	} from "phosphor-svelte";

	let data = $state<JournalResponse | null>(null);
	let loading = $state(false);
	let busy = $state(false);
	let error = $state<string | null>(null);
	let actionMsg = $state<string | null>(null);

	async function load() {
		loading = true;
		error = null;
		try {
			data = await journal();
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
		}
	}

	onMount(load);

	async function doScan() {
		busy = true;
		actionMsg = null;
		try {
			const r = await scanJournal({ scanner: "swing", model: "gbt", direction: "long" });
			actionMsg = `Logged ${r.logged} signal(s).`;
			await load();
		} catch (e) {
			actionMsg = e instanceof Error ? e.message : String(e);
		} finally {
			busy = false;
		}
	}

	async function doResolve() {
		busy = true;
		actionMsg = null;
		try {
			const r = await resolveJournal("swing");
			actionMsg = `Resolved ${r.resolved} of ${r.total}.`;
			await load();
		} catch (e) {
			actionMsg = e instanceof Error ? e.message : String(e);
		} finally {
			busy = false;
		}
	}

	async function exportJournal(format: "csv" | "xlsx") {
		if (!data?.entries.length) return;
		try {
			await downloadExport("journal", format, {
				journal: data.entries as unknown as Record<string, unknown>[]
			});
		} catch (e) {
			actionMsg = e instanceof Error ? e.message : String(e);
		}
	}

	const sign = (x: number) => `${x >= 0 ? "+" : ""}${x.toFixed(3)}`;
	const hasEntries = $derived(!!data?.entries.length);
</script>

<section class="journal">
	<div class="bar">
		<h2><NotebookIcon weight="duotone" size="1.1em" /> Live forward journal</h2>
		<div class="actions">
			<button onclick={doScan} disabled={busy}>
				<LightningIcon size={14} weight="bold" /> Scan
			</button>
			<button onclick={doResolve} disabled={busy}>
				<CheckCircleIcon size={14} weight="bold" /> Resolve
			</button>
			<button onclick={() => exportJournal("csv")} disabled={!hasEntries}>
				<FileCsvIcon size={14} /> CSV
			</button>
			<button onclick={() => exportJournal("xlsx")} disabled={!hasEntries}>
				<FileXlsIcon size={14} /> XLSX
			</button>
			<button onclick={load} disabled={busy || loading} aria-label="Refresh">
				<ArrowsClockwiseIcon size={14} weight="bold" />
			</button>
		</div>
	</div>

	{#if actionMsg}
		<p class="action-msg">{actionMsg}</p>
	{/if}

	{#if error}
		<p class="error">Journal unavailable: {error}</p>
	{:else if data}
		<p class="summary">
			<CurrencyDollarIcon size={15} />
			<b>{data.summary.open}</b> open · <b>{data.summary.resolved}</b> resolved · realized
			<b>{sign(data.summary.realized_mean_r)}R</b> (hit {(data.summary.realized_hit_rate * 100).toFixed(0)}%)
			· validated <b>{sign(data.summary.validated_edge_r)}R</b>
		</p>
		{#if data.entries.length}
			<table>
				<thead>
					<tr>
						<th>Symbol</th>
						<th>Dir</th>
						<th class="num">Entry</th>
						<th class="num">Stop</th>
						<th class="num">Target</th>
						<th class="num">Prob</th>
						<th>Status</th>
						<th class="num">Realized R</th>
						<th>Exit</th>
					</tr>
				</thead>
				<tbody>
					{#each data.entries as e, i (i)}
						<tr>
							<td>{e.symbol}</td>
							<td>
								<span class="dir {e.direction}">
									{#if e.direction === "long"}
										<TrendUpIcon size={12} weight="bold" />
									{:else}
										<TrendDownIcon size={12} weight="bold" />
									{/if}
									{e.direction}
								</span>
							</td>
							<td class="num">{e.entry.toFixed(2)}</td>
							<td class="num">{e.stop.toFixed(2)}</td>
							<td class="num">{e.target.toFixed(2)}</td>
							<td class="num">{(e.probability * 100).toFixed(0)}%</td>
							<td>
								<span class="status {e.status}">
									{#if e.status === "resolved"}
										<CheckCircleIcon size={12} weight="fill" />
									{:else}
										<ClockIcon size={12} />
									{/if}
									{e.status}
								</span>
							</td>
							<td class="num">{e.realized_r === undefined ? "—" : sign(e.realized_r)}</td>
							<td>{e.exit_reason ?? "—"}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		{:else}
			<p class="muted">No signals logged yet. Press <b>Scan</b> to log today's signals.</p>
		{/if}
	{:else}
		<p class="muted">Loading…</p>
	{/if}
</section>

<style>
	.journal {
		display: flex;
		flex-direction: column;
		gap: 0.6rem;
	}
	.bar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 1rem;
		flex-wrap: wrap;
	}
	.bar h2 {
		margin: 0;
		font-size: 1.1rem;
		display: flex;
		align-items: center;
		gap: 0.4rem;
	}
	.bar h2 :global(svg) {
		color: var(--accent);
	}
	.actions {
		display: flex;
		gap: 0.4rem;
		flex-wrap: wrap;
	}
	button {
		background: var(--bg-panel);
		color: var(--fg);
		border: 1px solid var(--border);
		border-radius: 6px;
		padding: 0.3rem 0.7rem;
		cursor: pointer;
		display: inline-flex;
		align-items: center;
		gap: 0.35rem;
		font-size: 0.85rem;
	}
	button:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	.action-msg {
		margin: 0;
		font-size: 0.8rem;
		color: var(--accent);
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
		padding: 0.35rem 0.6rem;
		border-bottom: 1px solid var(--border);
	}
	th.num,
	td.num {
		text-align: right;
	}
	.dir {
		text-transform: uppercase;
		font-size: 0.72rem;
		display: inline-flex;
		align-items: center;
		gap: 0.25rem;
	}
	.dir.long {
		color: var(--up);
	}
	.dir.short {
		color: var(--down);
	}
	.status {
		display: inline-flex;
		align-items: center;
		gap: 0.25rem;
	}
	.status.open {
		color: var(--accent);
	}
	.status.resolved {
		color: var(--muted);
	}
	.muted {
		color: var(--muted);
	}
	.error {
		color: var(--down);
	}
</style>
