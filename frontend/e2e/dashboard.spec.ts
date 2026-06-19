import { expect, test } from "@playwright/test";

// The /screen API is mocked so the test is deterministic and offline — it asserts
// the click -> render flow, not the engine (which has its own ~300 tests).

const ONE_SIGNAL = {
	summary: { configs_evaluated: 2, survived: 1, n_signals: 1 },
	signals: [
		{
			symbol: "TSLA",
			timestamp: "2026-06-18T10:00:00",
			direction: "long",
			event_type: "vwap_reclaim",
			entry: 100,
			stop: 98,
			target: 104,
			atr: 2,
			rr: 2,
			probability: 0.71,
			oos_edge_r: 0.42,
			p_value_fdr: 0.03,
			oos_auc: 0.66,
			decay: 0.05,
			n_events: 220,
			n_signals: 18,
			bracket: "reversal"
		}
	],
	reports: [
		{
			symbol: "TSLA",
			n_events: 220,
			n_signals: 18,
			oos_edge_r: 0.42,
			oos_auc: 0.66,
			p_value: 0.02,
			p_value_fdr: 0.03,
			decay: 0.05
		}
	]
};

const EMPTY = { summary: { configs_evaluated: 4, survived: 0, n_signals: 0 }, signals: [], reports: [] };

test("run screen renders a validated signal row with bracket geometry", async ({ page }) => {
	await page.route("**/screen", (route) => route.fulfill({ json: ONE_SIGNAL }));
	await page.goto("/");
	await expect(page.getByRole("heading", { name: /Trading Engine/ })).toBeVisible();

	await page.getByRole("button", { name: /Run screen/ }).click();

	await expect(page.getByText("1 signals")).toBeVisible();
	const row = page.locator("table.signals tbody tr").first();
	await expect(row.getByRole("button", { name: "TSLA" })).toBeVisible();
	await expect(row).toContainText("long");
	await expect(row).toContainText("100.00"); // entry
	await expect(row).toContainText("98.00"); // stop
	await expect(row).toContainText("104.00"); // target
});

test("empty screen shows the honest no-signals message", async ({ page }) => {
	await page.route("**/screen", (route) => route.fulfill({ json: EMPTY }));
	await page.goto("/");
	await page.getByRole("button", { name: /Run screen/ }).click();
	await expect(page.getByText(/no significant edge survived/i)).toBeVisible();
});

test("scanner selector switches to swing and hides intraday-only controls", async ({ page }) => {
	await page.goto("/");
	await expect(page.getByText("Timeframe")).toBeVisible(); // intraday default
	await page.getByLabel("Scanner").selectOption("swing");
	await expect(page.getByText("Timeframe")).toHaveCount(0); // hidden for swing
});
