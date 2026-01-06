import Papa from 'papaparse';
import { CalibrationDataPoint, CalibrationMetrics } from '@/types/calibration';

export function parseCalibrationCSV(csvContent: string): CalibrationDataPoint[] {
  const result = Papa.parse<CalibrationDataPoint>(csvContent, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });

  if (result.errors.length > 0) {
    throw new Error(`CSV parsing error: ${result.errors[0].message}`);
  }

  return result.data.filter(row => 
    row.epoch !== undefined && 
    row.loss !== undefined && 
    row.target_name !== undefined
  ).map(row => ({
    ...row,
    // Replace infinite values with a large finite number for better handling
    loss: isFinite(row.loss) ? row.loss : (row.loss > 0 ? 1e10 : -1e10),
    error: isFinite(row.error) ? row.error : (row.error > 0 ? 1e10 : -1e10),
    abs_error: isFinite(row.abs_error) ? row.abs_error : 1e10,
    rel_abs_error: isFinite(row.rel_abs_error) ? row.rel_abs_error : 1e10,
  }));
}

export function getCalibrationMetrics(data: CalibrationDataPoint[]): CalibrationMetrics {
  if (data.length === 0) {
    return {
      epochCount: 0,
      targetNames: [],
      finalLoss: 0,
    };
  }

  const epochs = [...new Set(data.map(d => d.epoch))].sort((a, b) => a - b);
  const targetNames = [...new Set(data.map(d => d.target_name))];
  const finalEpoch = epochs.length > 0 ? epochs[epochs.length - 1] : 0;
  const finalLoss = data.find(d => d.epoch === finalEpoch)?.loss || 0;

  // Find convergence epoch (when loss stops decreasing significantly)
  let convergenceEpoch: number | undefined;
  const lossByEpoch = epochs.map(epoch => {
    const epochData = data.find(d => d.epoch === epoch);
    return { epoch, loss: epochData?.loss || 0 };
  });

  for (let i = 1; i < lossByEpoch.length; i++) {
    const currentLoss = lossByEpoch[i].loss;
    const prevLoss = lossByEpoch[i - 1].loss;
    
    // Handle cases where loss values might be very large or infinite
    if (!isFinite(currentLoss) || !isFinite(prevLoss) || prevLoss === 0) {
      continue;
    }
    
    const improvement = (prevLoss - currentLoss) / prevLoss;
    
    if (improvement < 0.001) { // Less than 0.1% improvement
      convergenceEpoch = lossByEpoch[i].epoch;
      break;
    }
  }

  return {
    epochCount: epochs.length,
    targetNames,
    finalLoss,
    convergenceEpoch,
  };
}
