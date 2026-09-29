import { CalibrationDataPoint } from '@/types/calibration';

/**
 * Distinct epochs present in any of the datasets, newest first.
 */
export function getEpochsNewestFirst(...datasets: CalibrationDataPoint[][]): number[] {
  const epochs = new Set<number>();
  datasets.forEach(data => data.forEach(point => epochs.add(point.epoch)));
  return Array.from(epochs).sort((a, b) => b - a);
}

/**
 * Epoch a chart should display: the one the user picked, or the newest
 * available epoch until they pick one. Null only when there are no epochs
 * and nothing was picked.
 */
export function resolveSelectedEpoch(
  chosenEpoch: number | null,
  epochsNewestFirst: number[]
): number | null {
  return chosenEpoch ?? epochsNewestFirst[0] ?? null;
}
