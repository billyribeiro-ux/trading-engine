<script lang="ts">
	import favicon from "$lib/assets/favicon.svg";

	let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>

{@render children()}

<style>
	:global(:root) {
		--bg: #0b0f17;
		--bg-panel: #121826;
		--fg: #e6edf3;
		--fg-soft: #c2ccd6;
		--muted: #8b97a7;
		--border: #243044;
		--accent: #4cc2ff;
		--up: #3fb950;
		--down: #f85149;
		color-scheme: dark;
	}
	:global(body) {
		margin: 0;
		background: var(--bg);
		color: var(--fg);
		font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
		line-height: 1.5;
	}
	:global(*) {
		box-sizing: border-box;
	}
</style>
