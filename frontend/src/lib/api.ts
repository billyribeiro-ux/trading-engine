// Typed client for the trading-engine FastAPI backend.
// Base URL from PUBLIC_API_BASE (set in .env), defaulting to the local dev server.

import { env } from "$env/dynamic/public";
import type { Dissection, ScreenRequest, ScreenResponse } from "./types";

const BASE = (env.PUBLIC_API_BASE ?? "http://127.0.0.1:8000").replace(/\/$/, "");

async function parse<T>(res: Response): Promise<T> {
	if (!res.ok) {
		let detail = `${res.status} ${res.statusText}`;
		try {
			const body = await res.json();
			if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
		} catch {
			// non-JSON error body; keep the status line
		}
		throw new Error(detail);
	}
	return res.json() as Promise<T>;
}

export async function screen(req: ScreenRequest): Promise<ScreenResponse> {
	const res = await fetch(`${BASE}/screen`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(req)
	});
	return parse<ScreenResponse>(res);
}

export async function dissect(symbol: string, timeframe: string, date?: string): Promise<Dissection> {
	const params = new URLSearchParams({ timeframe });
	if (date) params.set("date", date);
	const res = await fetch(`${BASE}/dissect/${encodeURIComponent(symbol)}?${params}`);
	return parse<Dissection>(res);
}

export async function capabilities(): Promise<{ tier: string; rate_limit_per_min: number }> {
	return parse(await fetch(`${BASE}/capabilities`));
}
