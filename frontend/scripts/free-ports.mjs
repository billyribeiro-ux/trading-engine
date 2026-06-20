// Free the dev ports BEFORE starting, so `pnpm dev(:all)` can never fail with a
// zombie process / "address already in use". Kills whatever holds each port and
// waits until it's actually released. Ports come from argv (default 8000 + 5173).
import { execSync } from "node:child_process";

const ports = process.argv.slice(2).map(Number).filter(Boolean);
const targets = ports.length ? ports : [8000, 5173];

const sleep = (ms) => Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);

function pidsOn(port) {
	try {
		const out = execSync(`lsof -ti tcp:${port}`, {
			stdio: ["ignore", "pipe", "ignore"]
		})
			.toString()
			.trim();
		return out ? out.split("\n").filter(Boolean) : [];
	} catch {
		return []; // nothing listening
	}
}

for (const port of targets) {
	const pids = pidsOn(port);
	if (!pids.length) continue;
	for (const pid of pids) {
		try {
			process.kill(Number(pid), "SIGKILL");
		} catch {
			/* already gone */
		}
	}
	const start = Date.now();
	while (pidsOn(port).length && Date.now() - start < 3000) sleep(100);
	console.log(`[free-ports] :${port} cleared (was ${pids.join(", ")})`);
}
