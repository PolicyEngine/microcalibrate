'use client';

import { useMemo, useState } from 'react';
import { CalibrationDataPoint, ValidationDataPoint } from '@/types/calibration';
import { parseTargetName, variableKey, FIPS_TO_STATE } from '@/utils/targetNameParser';
import { globMatch } from '@/utils/globMatch';

interface Props {
  calibrationData: CalibrationDataPoint[];
  validationData: ValidationDataPoint[];
}

interface RollupRow {
  state: string;
  stateFips: number;
  variable: string;
  nDistricts: number;
  districtXwSum: number;
  stateSimValue: number | null;
  stateTarget: number | null;
  gap: number | null;
  gapPct: number | null;
}

type SortKey = keyof RollupRow;

function formatNumber(n: number): string {
  if (!isFinite(n)) return 'Inf';
  if (Math.abs(n) >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (Math.abs(n) >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (Math.abs(n) >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toFixed(2);
}

export default function StateRollupComparison({ calibrationData, validationData }: Props) {
  const [search, setSearch] = useState('');
  const [sortKey, setSortKey] = useState<SortKey>('gapPct');
  const [sortAsc, setSortAsc] = useState(false);

  const rows = useMemo(() => {
    const maxEpoch = Math.max(...calibrationData.map(d => d.epoch));
    const finalCal = calibrationData.filter(d => d.epoch === maxEpoch);

    // Group district-level cal data by (stateFips, variableKey)
    const districtSums = new Map<string, { sum: number; count: number; stateFips: number; variable: string }>();
    for (const d of finalCal) {
      const parsed = parseTargetName(d.target_name);
      if (parsed.geoLevel !== 'district' || parsed.stateFips === null) continue;
      const vk = variableKey(parsed);
      const key = `${parsed.stateFips}|${vk}`;
      const entry = districtSums.get(key);
      if (entry) {
        entry.sum += d.estimate;
        entry.count++;
      } else {
        districtSums.set(key, { sum: d.estimate, count: 1, stateFips: parsed.stateFips, variable: vk });
      }
    }

    // Build lookup for state-level validation data
    const stateValMap = new Map<string, ValidationDataPoint>();
    for (const v of validationData) {
      const parsed = parseTargetName(v.target_name);
      if (parsed.geoLevel !== 'state' || parsed.stateFips === null) continue;
      const vk = variableKey(parsed);
      const key = `${parsed.stateFips}|${vk}`;
      stateValMap.set(key, v);
    }

    const result: RollupRow[] = [];
    for (const [key, ds] of districtSums) {
      const stateVal = stateValMap.get(key);
      const stateAbbr = FIPS_TO_STATE[ds.stateFips] || `FIPS ${ds.stateFips}`;
      const gap = stateVal ? ds.sum - stateVal.sim_value : null;
      const gapPct = (gap !== null && stateVal && stateVal.sim_value !== 0)
        ? Math.abs(gap) / Math.abs(stateVal.sim_value)
        : null;

      result.push({
        state: stateAbbr,
        stateFips: ds.stateFips,
        variable: ds.variable,
        nDistricts: ds.count,
        districtXwSum: ds.sum,
        stateSimValue: stateVal?.sim_value ?? null,
        stateTarget: stateVal?.target_value ?? null,
        gap,
        gapPct,
      });
    }

    return result;
  }, [calibrationData, validationData]);

  const filtered = useMemo(() => {
    if (!search.trim()) return rows;
    return rows.filter(r => {
      const label = `${r.state}/${r.variable}`;
      return globMatch(search, label) || globMatch(search, r.variable) || globMatch(search, r.state);
    });
  }, [rows, search]);

  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      const va = a[sortKey];
      const vb = b[sortKey];
      if (va === null && vb === null) return 0;
      if (va === null) return 1;
      if (vb === null) return -1;
      if (typeof va === 'number' && typeof vb === 'number') {
        return sortAsc ? va - vb : vb - va;
      }
      return sortAsc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
    });
  }, [filtered, sortKey, sortAsc]);

  const handleSort = (key: SortKey) => {
    if (key === sortKey) setSortAsc(!sortAsc);
    else { setSortKey(key); setSortAsc(false); }
  };

  const nLargeGap = rows.filter(r => r.gapPct !== null && r.gapPct > 0.10).length;

  if (rows.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-2">State rollup comparison</h2>
        <p className="text-gray-500 text-sm">
          No district-level calibration data found to compare against state-level validation.
          Ensure calibration targets use <code>cd_XXXX/</code> prefixes and validation has <code>state_XX/</code> targets.
        </p>
      </div>
    );
  }

  const bannerColor = nLargeGap === 0
    ? 'bg-green-50 border-green-200 text-green-800'
    : nLargeGap < 10
      ? 'bg-yellow-50 border-yellow-200 text-yellow-800'
      : 'bg-red-50 border-red-200 text-red-800';

  const headerClass = "px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase cursor-pointer hover:text-gray-700";

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">State rollup comparison</h2>
      <p className="text-xs text-gray-500 mb-3">
        Sums district-level X*w estimates from calibration and compares to state-level sim.calculate values from validation.
      </p>

      <div className={`rounded-lg border p-3 mb-4 ${bannerColor}`}>
        <div className="font-medium text-sm">
          {nLargeGap === 0
            ? 'All state/variable combos have gaps ≤10%'
            : `${nLargeGap} state/variable combo${nLargeGap > 1 ? 's' : ''} have gaps >10%`}
        </div>
      </div>

      <div className="flex gap-3 mb-3">
        <input
          type="text"
          placeholder="Search... (* = wildcard)"
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="px-3 py-1.5 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 w-full max-w-md"
        />
        <span className="text-xs text-gray-500 self-center whitespace-nowrap">{filtered.length} of {rows.length} rows</span>
      </div>

      <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50 sticky top-0">
            <tr>
              <th className={headerClass} onClick={() => handleSort('state')}>State</th>
              <th className={headerClass} onClick={() => handleSort('variable')}>Variable</th>
              <th className={headerClass} onClick={() => handleSort('nDistricts')}># Districts</th>
              <th className={headerClass} onClick={() => handleSort('districtXwSum')}>Sum(District X*w)</th>
              <th className={headerClass} onClick={() => handleSort('stateSimValue')}>State Sim Value</th>
              <th className={headerClass} onClick={() => handleSort('stateTarget')}>State Target</th>
              <th className={headerClass} onClick={() => handleSort('gap')}>Gap</th>
              <th className={headerClass} onClick={() => handleSort('gapPct')}>Gap %</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {sorted.slice(0, 500).map((r, i) => {
              const gapColor = r.gapPct === null ? 'text-gray-400'
                : r.gapPct < 0.05 ? 'text-green-600'
                : r.gapPct < 0.20 ? 'text-yellow-600'
                : 'text-red-600';
              return (
                <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                  <td className="px-3 py-2 text-sm font-medium text-gray-900">{r.state}</td>
                  <td className="px-3 py-2 text-sm font-mono text-gray-700 max-w-xs truncate" title={r.variable}>{r.variable}</td>
                  <td className="px-3 py-2 text-sm text-gray-700 text-right">{r.nDistricts}</td>
                  <td className="px-3 py-2 text-sm text-gray-700 text-right">{formatNumber(r.districtXwSum)}</td>
                  <td className="px-3 py-2 text-sm text-gray-700 text-right">{r.stateSimValue !== null ? formatNumber(r.stateSimValue) : '—'}</td>
                  <td className="px-3 py-2 text-sm text-gray-700 text-right">{r.stateTarget !== null ? formatNumber(r.stateTarget) : '—'}</td>
                  <td className="px-3 py-2 text-sm text-right">
                    <span className={gapColor}>{r.gap !== null ? formatNumber(r.gap) : '—'}</span>
                  </td>
                  <td className="px-3 py-2 text-sm text-right">
                    <span className={`font-medium ${gapColor}`}>
                      {r.gapPct !== null ? `${(r.gapPct * 100).toFixed(1)}%` : '—'}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {sorted.length > 500 && (
          <p className="text-xs text-gray-400 p-2">Showing first 500 of {sorted.length} rows</p>
        )}
      </div>
    </div>
  );
}
