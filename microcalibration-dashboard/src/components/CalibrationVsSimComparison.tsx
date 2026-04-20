'use client';

import { useMemo, useState } from 'react';
import { CalibrationDataPoint, ValidationDataPoint } from '@/types/calibration';
import { globMatch } from '@/utils/globMatch';

interface CalibrationVsSimComparisonProps {
  calibrationData: CalibrationDataPoint[];
  validationData: ValidationDataPoint[];
}

function formatNumber(n: number): string {
  if (!isFinite(n)) return 'Inf';
  if (Math.abs(n) >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (Math.abs(n) >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (Math.abs(n) >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toFixed(2);
}

interface JoinedRow {
  target_name: string;
  xw_estimate: number;
  sim_value: number;
  target_value: number;
  xw_error: number;
  sim_error: number;
  gap: number;
}

type SortKey = keyof JoinedRow;

export default function CalibrationVsSimComparison({ calibrationData, validationData }: CalibrationVsSimComparisonProps) {
  const [filterVar, setFilterVar] = useState('');
  const [sortKey, setSortKey] = useState<SortKey>('target_name');
  const [sortAsc, setSortAsc] = useState(true);

  const joined = useMemo(() => {
    // Get final epoch calibration data
    const maxEpoch = Math.max(...calibrationData.map(d => d.epoch));
    const finalEpochData = calibrationData.filter(d => d.epoch === maxEpoch);

    const calByName = new Map<string, CalibrationDataPoint>();
    for (const d of finalEpochData) {
      calByName.set(d.target_name, d);
    }

    const rows: JoinedRow[] = [];
    for (const v of validationData) {
      const cal = calByName.get(v.target_name);
      if (!cal) continue;

      const xw_error = cal.estimate - cal.target;
      const sim_error = v.sim_value - v.target_value;
      rows.push({
        target_name: v.target_name,
        xw_estimate: cal.estimate,
        sim_value: v.sim_value,
        target_value: v.target_value,
        xw_error,
        sim_error,
        gap: sim_error - xw_error,
      });
    }

    return rows;
  }, [calibrationData, validationData]);

  const filtered = useMemo(() => {
    if (!filterVar) return joined;
    return joined.filter(r => globMatch(filterVar, r.target_name));
  }, [joined, filterVar]);

  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      const va = a[sortKey];
      const vb = b[sortKey];
      if (typeof va === 'number' && typeof vb === 'number') {
        return sortAsc ? va - vb : vb - va;
      }
      const sa = String(va);
      const sb = String(vb);
      return sortAsc ? sa.localeCompare(sb) : sb.localeCompare(sa);
    });
  }, [filtered, sortKey, sortAsc]);

  const handleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(false);
    }
  };

  const regressions = useMemo(() => {
    const regressed: JoinedRow[] = [];
    for (const r of joined) {
      if (r.target_value === 0) continue;
      const xwRelErr = Math.abs(r.xw_error) / Math.abs(r.target_value);
      const simRelErr = Math.abs(r.sim_error) / Math.abs(r.target_value);
      if (simRelErr > xwRelErr + 0.10) {
        regressed.push(r);
      }
    }
    return regressed;
  }, [joined]);

  const worstRegressions = useMemo(() => {
    return [...regressions]
      .sort((a, b) => {
        const aRel = Math.abs(a.target_value) > 0 ? Math.abs(a.gap) / Math.abs(a.target_value) : 0;
        const bRel = Math.abs(b.target_value) > 0 ? Math.abs(b.gap) / Math.abs(b.target_value) : 0;
        return bRel - aRel;
      })
      .slice(0, 5);
  }, [regressions]);

  if (joined.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Calibration vs simulation</h2>
        <p className="text-gray-500">No matching target_names found between calibration and validation data. Ensure both CSVs use the same target_name format.</p>
      </div>
    );
  }

  const meanAbsGap = sorted.reduce((s, r) => s + Math.abs(r.gap), 0) / sorted.length;
  const nWorse = sorted.filter(r => Math.abs(r.sim_error) > Math.abs(r.xw_error)).length;
  const nRegressed10pct = regressions.length;

  const bannerColor = nRegressed10pct === 0
    ? 'bg-green-50 border-green-200 text-green-800'
    : nRegressed10pct < 5
      ? 'bg-yellow-50 border-yellow-200 text-yellow-800'
      : 'bg-red-50 border-red-200 text-red-800';

  const bannerLabel = nRegressed10pct === 0
    ? 'No significant regressions'
    : nRegressed10pct < 5
      ? `${nRegressed10pct} target(s) regressed >10pp`
      : `${nRegressed10pct} targets regressed >10pp`;

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-2">Calibration vs simulation comparison</h2>
      <p className="text-xs text-gray-500 mb-4">
        Compares the calibration optimizer&apos;s estimate (X*w) against what sim.calculate() actually produces for the same targets.
        The <strong>gap</strong> = sim_error &minus; xw_error, i.e. how much the simulation drifted from what calibration predicted.
        Small gaps (green) mean sim and X*w are close; large gaps (orange) indicate significant drift between the two.
      </p>

      {/* Regression severity banner */}
      <div className={`rounded-lg border p-3 mb-4 ${bannerColor}`}>
        <div className="font-medium text-sm">{bannerLabel}</div>
        {worstRegressions.length > 0 && (
          <ul className="mt-2 text-xs space-y-1">
            {worstRegressions.map((r, i) => {
              const relGap = Math.abs(r.target_value) > 0
                ? (Math.abs(r.gap) / Math.abs(r.target_value) * 100).toFixed(1)
                : '?';
              return (
                <li key={i} className="font-mono">
                  {r.target_name}: gap {formatNumber(r.gap)} ({relGap}% of target)
                </li>
              );
            })}
          </ul>
        )}
      </div>

      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="bg-gray-50 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-blue-600">{joined.length}</div>
          <div className="text-xs text-gray-500">Matched targets</div>
        </div>
        <div className="bg-gray-50 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-amber-600">{formatNumber(meanAbsGap)}</div>
          <div className="text-xs text-gray-500">Mean |gap|</div>
        </div>
        <div className="bg-gray-50 rounded-lg p-3 text-center">
          <div className={`text-2xl font-bold ${nWorse > joined.length / 2 ? 'text-red-600' : 'text-green-600'}`}>
            {nWorse}/{sorted.length}
          </div>
          <div className="text-xs text-gray-500">Sim worse than X*w</div>
        </div>
      </div>

      <input
        type="text"
        placeholder="Search... (* = wildcard)"
        value={filterVar}
        onChange={e => setFilterVar(e.target.value)}
        className="px-3 py-1.5 border border-gray-300 rounded-md text-sm mb-3 w-full max-w-md focus:outline-none focus:ring-2 focus:ring-blue-500"
      />

      <p className="text-xs text-gray-500 mb-2">{sorted.length} rows</p>

      <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50 sticky top-0">
            <tr>
              {([
                ['target_name', 'Target Name', 'text-left'],
                ['target_value', 'Target', 'text-right'],
                ['xw_estimate', 'X*w Est.', 'text-right'],
                ['sim_value', 'Sim Value', 'text-right'],
                ['xw_error', 'X*w Error', 'text-right'],
                ['sim_error', 'Sim Error', 'text-right'],
                ['gap', 'Gap', 'text-right'],
              ] as [SortKey, string, string][]).map(([key, label, align]) => (
                <th
                  key={key}
                  className={`px-3 py-2 ${align} text-xs font-medium text-gray-500 uppercase cursor-pointer hover:text-gray-700 select-none`}
                  onClick={() => handleSort(key)}
                >
                  {label} {sortKey === key ? (sortAsc ? '▲' : '▼') : ''}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {sorted.slice(0, 500).map((r, i) => (
              <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                <td className="px-3 py-2 text-sm font-mono text-gray-900 max-w-64" title={r.target_name}>
                  <div
                    className="overflow-x-auto whitespace-nowrap [&::-webkit-scrollbar]:hidden"
                    style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
                  >
                    {r.target_name}
                  </div>
                </td>
                <td className="px-3 py-2 text-sm text-gray-700 text-right">{formatNumber(r.target_value)}</td>
                <td className="px-3 py-2 text-sm text-gray-700 text-right">{formatNumber(r.xw_estimate)}</td>
                <td className="px-3 py-2 text-sm text-gray-700 text-right">{formatNumber(r.sim_value)}</td>
                <td className="px-3 py-2 text-sm text-right">
                  <span className={Math.abs(r.xw_error) / Math.abs(r.target_value) < 0.05 ? 'text-green-600' : 'text-orange-600'}>
                    {formatNumber(r.xw_error)}
                  </span>
                </td>
                <td className="px-3 py-2 text-sm text-right">
                  <span className={Math.abs(r.sim_error) / Math.abs(r.target_value) < 0.05 ? 'text-green-600' : 'text-orange-600'}>
                    {formatNumber(r.sim_error)}
                  </span>
                </td>
                <td className="px-3 py-2 text-sm text-right">
                  <span className={Math.abs(r.gap) / Math.abs(r.target_value) < 0.05 ? 'text-green-600' : 'text-orange-600'}>
                    {formatNumber(r.gap)}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
