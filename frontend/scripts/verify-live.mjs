// Hard-evidence smoke test against a RUNNING dashboard (api + web).
// Usage: node scripts/verify-live.mjs [url]   (default http://localhost:5173)
// Exits 0 only if: page loads, journal renders through the browser (CORS ok),
// a screen completes, and there are ZERO console/page/network errors.
import { chromium } from "@playwright/test";

const url = process.argv[2] ?? "http://localhost:5173";
const errors = [];
const browser = await chromium.launch();
const page = await browser.newPage();
page.on("console", (m) => m.type() === "error" && errors.push("console: " + m.text()));
page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
page.on("requestfailed", (r) =>
	errors.push("requestfailed: " + r.url() + " " + (r.failure()?.errorText ?? ""))
);
page.on("response", (r) => r.status() >= 500 && errors.push("HTTP " + r.status() + " " + r.url()));

const checks = [];
const ok = (name, cond, detail = "") => checks.push({ name, pass: !!cond, detail });

try {
	await page.goto(url, { waitUntil: "networkidle", timeout: 30000 });
	ok("page loads", true);

	// Journal must render its summary — proves GET /journal succeeded via CORS.
	const summary = page.locator("section.journal .summary");
	await summary.waitFor({ state: "visible", timeout: 15000 }).catch(() => {});
	const sText = (await summary.innerText().catch(() => "")) || "";
	ok("journal renders (CORS ok)", /open/.test(sText), sText.replace(/\s+/g, " ").slice(0, 90));

	// Run a screen and confirm it completes (signals OR the honest empty message).
	await page.fill("input[name=watchlist]", "AAPL, MSFT, NVDA").catch(() => {});
	await page.click('button[type="submit"]');
	await page
		.locator("table.signals, p.muted:has-text('per-symbol')")
		.first()
		.waitFor({ state: "visible", timeout: 30000 })
		.catch(() => {});
	const screenDone = await page
		.locator("table.signals, p.muted:has-text('per-symbol')")
		.first()
		.isVisible()
		.catch(() => false);
	ok("screen completes", screenDone);

	// Settings opens and reads status.
	await page.click("button.settings-btn").catch(() => {});
	const cfg = await page
		.locator("dialog .state")
		.innerText()
		.catch(() => "");
	ok("settings loads", /Configured|No key/.test(cfg), cfg.replace(/\s+/g, " ").slice(0, 60));
} catch (e) {
	ok("run", false, e.message);
}

ok("zero console/network errors", errors.length === 0);

console.log("\n=== LIVE VERIFICATION (" + url + ") ===");
for (const c of checks) console.log(`  ${c.pass ? "PASS" : "FAIL"}  ${c.name}${c.detail ? "  — " + c.detail : ""}`);
if (errors.length) {
	console.log("  errors:");
	for (const e of [...new Set(errors)]) console.log("    - " + e);
}
const allPass = checks.every((c) => c.pass);
console.log(allPass ? "\nALL GREEN ✅" : "\nFAILED ❌");
await browser.close();
process.exit(allPass ? 0 : 1);
