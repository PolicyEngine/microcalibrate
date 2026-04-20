import Papa from 'papaparse';
import { ValidationDataPoint } from '@/types/calibration';

export function isValidationCsv(csvContent: string): boolean {
  const firstLine = csvContent.split('\n')[0]?.toLowerCase() ?? '';
  return firstLine.includes('sim_value');
}

export function parseValidationCSV(csvContent: string): ValidationDataPoint[] {
  const result = Papa.parse<Record<string, string>>(csvContent, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });

  if (result.errors.length > 0) {
    throw new Error(`CSV parsing error: ${result.errors[0].message}`);
  }

  return result.data
    .filter(row => row.sim_value !== undefined && row.target_value !== undefined)
    .map(row => ({
      area_type: String(row.area_type ?? ''),
      area_id: String(row.area_id ?? ''),
      variable: String(row.variable ?? ''),
      target_name: String(row.target_name ?? ''),
      period: Number(row.period ?? 0),
      target_value: Number(row.target_value ?? 0),
      sim_value: Number(row.sim_value ?? 0),
      error: Number(row.error ?? 0),
      rel_error: isFinite(Number(row.rel_error)) ? Number(row.rel_error) : 1e10,
      abs_error: Number(row.abs_error ?? 0),
      rel_abs_error: isFinite(Number(row.rel_abs_error)) ? Number(row.rel_abs_error) : 1e10,
      sanity_check: String(row.sanity_check ?? 'PASS'),
      sanity_reason: String(row.sanity_reason ?? ''),
      in_training: row.in_training === 'True' || row.in_training === 'true' || row.in_training === true as unknown as string,
    }));
}
