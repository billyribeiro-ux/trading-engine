<script lang="ts">
	import { onMount } from "svelte";
	import { getSettings, saveSettings } from "$lib/api";
	import type { SettingsStatus } from "$lib/types";
	import {
		CheckCircleIcon,
		EyeIcon,
		EyeSlashIcon,
		GearSixIcon,
		KeyIcon,
		WarningCircleIcon,
		XIcon
	} from "phosphor-svelte";

	let { open = $bindable(false) }: { open?: boolean } = $props();

	let dialogEl: HTMLDialogElement | undefined = $state();
	let status = $state<SettingsStatus | null>(null);
	let keyInput = $state("");
	let reveal = $state(false);
	let saving = $state(false);
	let message = $state<{ kind: "ok" | "err"; text: string } | null>(null);

	async function loadStatus() {
		try {
			status = await getSettings();
		} catch {
			status = null;
		}
	}

	onMount(loadStatus); // status only changes when we save it (and after-save below)

	// Keep the native <dialog> in sync with the bound `open` prop — DOM only.
	$effect(() => {
		if (!dialogEl) return;
		if (open && !dialogEl.open) dialogEl.showModal();
		else if (!open && dialogEl.open) dialogEl.close();
	});

	function onClosed() {
		open = false;
		keyInput = "";
		reveal = false;
		message = null;
	}

	async function save() {
		const key = keyInput.trim();
		if (key.length < 8) {
			message = { kind: "err", text: "That key looks too short." };
			return;
		}
		saving = true;
		message = null;
		try {
			const res = await saveSettings(key);
			message = { kind: "ok", text: `Saved — FMP tier ${res.tier ?? "unknown"}.` };
			keyInput = "";
			await loadStatus();
		} catch (e) {
			message = { kind: "err", text: e instanceof Error ? e.message : String(e) };
		} finally {
			saving = false;
		}
	}
</script>

<dialog bind:this={dialogEl} onclose={onClosed} aria-label="Settings">
	<div class="panel">
		<header>
			<h2><GearSixIcon weight="duotone" /> Settings</h2>
			<button class="x" onclick={() => dialogEl?.close()} aria-label="Close settings">
				<XIcon weight="bold" />
			</button>
		</header>

		<section class="group">
			<h3><KeyIcon /> FMP API key</h3>
			{#if status?.configured}
				<p class="state ok">
					<CheckCircleIcon weight="fill" />
					Configured ({status.source}) — {status.masked}
				</p>
			{:else}
				<p class="state warn">
					<WarningCircleIcon weight="fill" />
					No key set — the engine can't fetch market data.
				</p>
			{/if}

			<div class="field">
				<input
					id="fmp-api-key"
					name="fmp_api_key"
					type={reveal ? "text" : "password"}
					bind:value={keyInput}
					placeholder="Paste your FMP API key"
					autocomplete="off"
					spellcheck="false"
				/>
				<button
					class="reveal"
					type="button"
					onclick={() => (reveal = !reveal)}
					aria-label={reveal ? "Hide key" : "Show key"}
				>
					{#if reveal}<EyeSlashIcon />{:else}<EyeIcon />{/if}
				</button>
			</div>

			<button class="save" onclick={save} disabled={saving || keyInput.trim().length < 8}>
				{saving ? "Validating…" : "Validate & save"}
			</button>

			{#if message}
				<p class="msg {message.kind}">{message.text}</p>
			{/if}
			<p class="hint">Stored locally (chmod 600), never committed. Validated against FMP on save.</p>
		</section>

		<p class="more">More settings coming soon.</p>
	</div>
</dialog>

<style>
	dialog {
		border: 1px solid var(--border);
		border-radius: 12px;
		background: var(--bg-panel);
		color: var(--fg);
		padding: 0;
		width: min(440px, 92vw);
		box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
	}
	dialog::backdrop {
		background: rgba(0, 0, 0, 0.55);
		backdrop-filter: blur(2px);
	}
	.panel {
		display: flex;
		flex-direction: column;
		gap: 1rem;
		padding: 1.25rem;
	}
	header {
		display: flex;
		align-items: center;
		justify-content: space-between;
	}
	header h2 {
		margin: 0;
		font-size: 1.2rem;
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}
	header h2 :global(svg) {
		color: var(--accent);
	}
	.x {
		background: none;
		border: none;
		color: var(--muted);
		cursor: pointer;
		display: inline-flex;
		padding: 0.25rem;
		border-radius: 6px;
	}
	.x:hover {
		color: var(--fg);
		background: var(--bg);
	}
	.group {
		display: flex;
		flex-direction: column;
		gap: 0.6rem;
		border: 1px solid var(--border);
		border-radius: 8px;
		padding: 0.9rem;
	}
	.group h3 {
		margin: 0;
		font-size: 0.95rem;
		display: flex;
		align-items: center;
		gap: 0.4rem;
	}
	.state {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		font-size: 0.85rem;
		margin: 0;
	}
	.state.ok {
		color: var(--up);
	}
	.state.warn {
		color: var(--down);
	}
	.field {
		display: flex;
		gap: 0.4rem;
	}
	.field input {
		flex: 1;
		background: var(--bg);
		color: var(--fg);
		border: 1px solid var(--border);
		border-radius: 6px;
		padding: 0.5rem 0.6rem;
		font: inherit;
	}
	.reveal {
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: 6px;
		color: var(--muted);
		cursor: pointer;
		display: inline-flex;
		align-items: center;
		padding: 0 0.5rem;
	}
	.save {
		background: var(--accent);
		color: #04111f;
		border: none;
		border-radius: 6px;
		padding: 0.55rem 1rem;
		font-weight: 600;
		cursor: pointer;
	}
	.save:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	.msg {
		margin: 0;
		font-size: 0.85rem;
	}
	.msg.ok {
		color: var(--up);
	}
	.msg.err {
		color: var(--down);
	}
	.hint {
		margin: 0;
		font-size: 0.75rem;
		color: var(--muted);
	}
	.more {
		margin: 0;
		font-size: 0.75rem;
		color: var(--muted);
		text-align: center;
	}
</style>
