// Shared types mirroring the FastAPI responses (engine/api/serialize.py).

export type Direction = "long" | "short";

export interface SignalOut {
	symbol: string;
	timestamp: string;
	direction: Direction;
	event_type: string;
	entry: number;
	stop: number;
	target: number;
	atr: number;
	rr: number;
	probability: number;
	oos_edge_r: number;
	p_value_fdr: number;
	oos_auc: number;
	decay: number;
	n_events: number;
	n_signals: number;
	bracket: string;
}

export interface ReportOut {
	symbol: string;
	n_events: number;
	n_signals: number;
	oos_edge_r: number;
	oos_auc: number;
	p_value: number;
	p_value_fdr: number;
	decay: number;
}

export interface ScreenResponse {
	summary: { configs_evaluated: number; survived: number; n_signals: number };
	signals: SignalOut[];
	reports: ReportOut[];
}

export interface ScreenRequest {
	symbols: string[];
	scanner: string; // "intraday" | "swing" | "portfolio"
	timeframe: string;
	lookback_days: number;
	style: string;
	proba_threshold?: number;
	fdr?: number;
	min_edge_r?: number;
}

export interface Leg {
	direction: string;
	start_time: string;
	end_time: string;
	duration_min: number;
	start_price: number;
	end_price: number;
	magnitude: number;
}

export interface StructureLeg extends Leg {
	n: number;
	sub_scale: number | null;
	sub_legs: Leg[];
}

export interface LegRole {
	n: number;
	direction: string;
	magnitude_atr: number;
	vwap_start: string;
	vwap_end: string;
	roles: string[];
}

export interface VwapEvent {
	type: string;
	count: number;
	first_time: string;
	last_time: string;
	outcome_atr: number;
}

export interface LevelEvent {
	time: string;
	kind: string;
	level: number;
	held: boolean;
	outcome_atr: number;
}

export interface DissectionHeader {
	open: number;
	high: number;
	low: number;
	close: number;
	range: number;
	range_pct: number;
	range_atr: number;
	vwap_close: number;
	bias: string;
	high_time: string;
	low_time: string;
	scale_atr: number | null;
}

export interface Dissection {
	symbol: string;
	date: string;
	header: DissectionHeader;
	structure: StructureLeg[];
	leg_roles: LegRole[];
	vwap_map: VwapEvent[];
	levels: LevelEvent[];
	read: string;
	consistent: boolean;
}
