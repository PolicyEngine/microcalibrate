export interface CalibrationDataPoint {
  epoch: number;
  loss: number;
  target_name: string;
  target: number;
  estimate: number;
  error: number;
  abs_error: number;
  rel_abs_error: number;
}

export interface CalibrationMetrics {
  epochCount: number;
  targetNames: string[];
  finalLoss: number;
  convergenceEpoch?: number;
}
