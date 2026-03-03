export interface CalibrationDataPoint {
  epoch: number;
  loss: number;
  target_name: string;
  target: number;
  estimate: number;
  error: number;
  abs_error: number;
  rel_abs_error: number;
  rel_error?: number;
  achievable?: boolean;
}

export interface CalibrationMetrics {
  epochCount: number;
  targetNames: string[];
  finalLoss: number;
  convergenceEpoch?: number;
}

export interface ValidationDataPoint {
  area_type: string;
  area_id: string;
  variable: string;
  target_name: string;
  period: number;
  target_value: number;
  sim_value: number;
  error: number;
  rel_error: number;
  abs_error: number;
  rel_abs_error: number;
  sanity_check: string;
  sanity_reason: string;
  in_training: boolean;
}
