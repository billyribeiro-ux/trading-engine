import { defineConfig, devices } from "@playwright/test";

// E2E against the production build (adapter-node preview). The API is mocked in
// the specs (page.route), so these run offline and deterministically.
export default defineConfig({
	testDir: "e2e",
	timeout: 30_000,
	expect: { timeout: 7_000 },
	fullyParallel: true,
	use: { baseURL: "http://localhost:4173" },
	projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
	webServer: {
		command: "pnpm run build && pnpm run preview --port 4173",
		url: "http://localhost:4173",
		timeout: 120_000,
		reuseExistingServer: !process.env.CI
	}
});
