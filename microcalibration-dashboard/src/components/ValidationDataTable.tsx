'use client';

import { useState, useMemo } from 'react';
import { ValidationDataPoint } from '@/types/calibration';
import { globMatch } from '@/utils/globMatch';

interface ValidationDataTableProps {
  data: ValidationDataPoint[];
}

type SortKey = 'area_id' | 'variable' | 'target_value' | 'sim_value' | 'rel_abs_error' | 'sanity_check';

function formatNumber(n: number): string {
  if (!isFinite(n)) return 'Inf';
  if (Math.abs(n) >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (Math.abs(n) >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toFixed(2);
}

export default function ValidationDataTable({ data }: ValidationDataTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>('rel_abs_error');
  const [sortAsc, setSortAsc] = useState(false);
  const [filterVar, setFilterVar] = useState('');
  const [filterAreaType, setFilterAreaType] = useState('');
  const [maxError, setMaxError] = useState<number | null>(null);
  const [showTrainingOnly, setShowTrainingOnly] = useState<boolean | null>(null);

  const areaTypes = useMemo(() => [...new Set(data.map(d => d.area_type))], [data]);

  const filtered = useMemo(() => {
    let result = data;
    if (filterVar) {
      result = result.filter(d => globMatch(filterVar, d.target_name) || globMatch(filterVar, d.variable));
    }
    if (filterAreaType) {
      result = result.filter(d => d.area_type === filterAreaType);
    }
    if (maxError !== null) {
      result = result.filter(d => d.rel_abs_error <= maxError);
    }
    if (showTrainingOnly === true) {
      result = result.filter(d => d.in_training);
    } else if (showTrainingOnly === false) {
      result = result.filter(d => !d.in_training);
    }
    return result;
  }, [data, filterVar, filterAreaType, maxError, showTrainingOnly]);

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

  const headerClass = "px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase cursor-pointer hover:text-gray-700";

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">Validation results</h2>

      <div className="flex flex-wrap gap-3 mb-4">
        <input
          type="text"
          placeholder="Search... (* = wildcard)"
          value={filterVar}
          onChange={e => setFilterVar(e.target.value)}
          className="px-3 py-1.5 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        {areaTypes.length > 1 && (
          <select
            value={filterAreaType}
            onChange={e => setFilterAreaType(e.target.value)}
            className="px-3 py-1.5 border border-gray-300 rounded-md text-sm"
          >
            <option value="">All area types</option>
            {areaTypes.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
        )}
        <select
          value={maxError === null ? '' : String(maxError)}
          onChange={e => setMaxError(e.target.value ? Number(e.target.value) : null)}
          className="px-3 py-1.5 border border-gray-300 rounded-md text-sm"
        >
          <option value="">All errors</option>
          <option value="0.05">&le; 5%</option>
          <option value="0.10">&le; 10%</option>
          <option value="0.20">&le; 20%</option>
          <option value="0.50">&le; 50%</option>
          <option value="1.00">&le; 100%</option>
        </select>
        <select
          value={showTrainingOnly === null ? '' : showTrainingOnly ? 'train' : 'oos'}
          onChange={e => {
            const v = e.target.value;
            setShowTrainingOnly(v === '' ? null : v === 'train');
          }}
          className="px-3 py-1.5 border border-gray-300 rounded-md text-sm"
        >
          <option value="">All targets</option>
          <option value="train">In-training</option>
          <option value="oos">Out-of-sample</option>
        </select>
      </div>

      <p className="text-xs text-gray-500 mb-2">{sorted.length} of {data.length} rows shown</p>

      <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50 sticky top-0">
            <tr>
              <th className={headerClass} onClick={() => handleSort('area_id')}>Area</th>
              <th className={headerClass} onClick={() => handleSort('variable')}>Target Definition</th>
              <th className={headerClass} onClick={() => handleSort('target_value')}>Target</th>
              <th className={headerClass} onClick={() => handleSort('sim_value')}>Sim Value</th>
              <th className={headerClass} onClick={() => handleSort('rel_abs_error')}>Rel Abs Error</th>
              <th className={headerClass} onClick={() => handleSort('sanity_check')}>Sanity</th>
              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Training</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {sorted.slice(0, 500).map((d, i) => (
              <tr key={i} className={d.sanity_check === 'FAIL' ? 'bg-red-50' : i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                <td className="px-3 py-2 text-sm text-gray-700">{d.area_id}</td>
                <td className="px-3 py-2 text-sm font-mono text-gray-900 max-w-md truncate" title={d.target_name}>{d.target_name}</td>
                <td className="px-3 py-2 text-sm text-gray-700 text-right">{formatNumber(d.target_value)}</td>
                <td className="px-3 py-2 text-sm text-gray-700 text-right">{formatNumber(d.sim_value)}</td>
                <td className="px-3 py-2 text-sm text-right">
                  <span className={
                    d.rel_abs_error < 0.05 ? 'text-green-600 font-medium' :
                    d.rel_abs_error < 0.20 ? 'text-yellow-600' :
                    'text-red-600 font-medium'
                  }>
                    {isFinite(d.rel_abs_error) ? `${(d.rel_abs_error * 100).toFixed(1)}%` : 'Inf'}
                  </span>
                </td>
                <td className="px-3 py-2 text-sm">
                  {d.sanity_check === 'FAIL' ? (
                    <span className="text-red-600 font-medium" title={d.sanity_reason}>FAIL</span>
                  ) : (
                    <span className="text-green-600">PASS</span>
                  )}
                </td>
                <td className="px-3 py-2 text-sm text-center">
                  {d.in_training ? (
                    <span className="inline-block w-2 h-2 rounded-full bg-blue-500" title="In training" />
                  ) : (
                    <span className="inline-block w-2 h-2 rounded-full bg-gray-300" title="Out of sample" />
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {sorted.length > 500 && (
          <p className="text-xs text-gray-400 p-2">Showing first 500 of {sorted.length} rows</p>
        )}
      </div>
    </div>
  );
}
