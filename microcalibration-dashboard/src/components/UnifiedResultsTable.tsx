'use client';

import { useMemo, useState } from 'react';
import { CalibrationDataPoint, ValidationDataPoint } from '@/types/calibration';
import { parseTargetName } from '@/utils/targetNameParser';
import { globMatch } from '@/utils/globMatch';
import { compareTargetNames } from '@/utils/targetOrdering';

interface Props {
  calibrationData: CalibrationDataPoint[];
  validationData: ValidationDataPoint[];
}

interface UnifiedRow {
  source: 'cal' | 'val';
  geo: string;
  geoLevel: string;
  variable: string;
  constraint: string;
  target_name: string;
  target_value: number;
  estimate: number;
  rel_abs_error: number;
}

type SortKey = keyof UnifiedRow;
type SourceFilter = 'all' | 'cal' | 'val';
type GeoFilter = 'all' | 'national' | 'district' | 'state';

function formatNumber(n: number): string {
  if (!isFinite(n)) return 'Inf';
  if (Math.abs(n) >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (Math.abs(n) >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (Math.abs(n) >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toFixed(2);
}

export default function UnifiedResultsTable({ calibrationData, validationData }: Props) {
  const [search, setSearch] = useState('');
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>('all');
  const [geoFilter, setGeoFilter] = useState<GeoFilter>('all');
  const [variableFilter, setVariableFilter] = useState('');
  const [sortKey, setSortKey] = useState<SortKey>('target_name');
  const [sortAsc, setSortAsc] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 50;

  const allRows = useMemo(() => {
    const rows: UnifiedRow[] = [];

    // Calibration: final epoch only
    if (calibrationData.length > 0) {
      const maxEpoch = Math.max(...calibrationData.map(d => d.epoch));
      const finalCal = calibrationData.filter(d => d.epoch === maxEpoch);
      for (const d of finalCal) {
        const parsed = parseTargetName(d.target_name);
        rows.push({
          source: 'cal',
          geo: parsed.geoPrefix,
          geoLevel: parsed.geoLevel,
          variable: parsed.variable,
          constraint: parsed.constraint,
          target_name: d.target_name,
          target_value: d.target,
          estimate: d.estimate,
          rel_abs_error: d.rel_abs_error,
        });
      }
    }

    // Validation
    for (const d of validationData) {
      const parsed = parseTargetName(d.target_name);
      rows.push({
        source: 'val',
        geo: parsed.geoPrefix,
        geoLevel: parsed.geoLevel,
        variable: parsed.variable,
        constraint: parsed.constraint,
        target_name: d.target_name,
        target_value: d.target_value,
        estimate: d.sim_value,
        rel_abs_error: d.rel_abs_error,
      });
    }

    return rows;
  }, [calibrationData, validationData]);

  const variables = useMemo(() => {
    const s = new Set(allRows.map(r => r.variable));
    return Array.from(s).sort();
  }, [allRows]);

  const filtered = useMemo(() => {
    let result = allRows;
    if (sourceFilter !== 'all') {
      result = result.filter(r => r.source === sourceFilter);
    }
    if (geoFilter !== 'all') {
      result = result.filter(r => r.geoLevel === geoFilter);
    }
    if (variableFilter) {
      result = result.filter(r => r.variable === variableFilter);
    }
    if (search.trim()) {
      result = result.filter(r => globMatch(search, r.target_name));
    }
    return result;
  }, [allRows, sourceFilter, geoFilter, variableFilter, search]);

  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      if (sortKey === 'target_name') {
        const result = compareTargetNames(a.target_name, b.target_name);
        return sortAsc ? result : -result;
      }
      const va = a[sortKey];
      const vb = b[sortKey];
      if (typeof va === 'number' && typeof vb === 'number') {
        return sortAsc ? va - vb : vb - va;
      }
      return sortAsc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
    });
  }, [filtered, sortKey, sortAsc]);

  const totalPages = Math.ceil(sorted.length / itemsPerPage);
  const paginated = useMemo(() => {
    const start = (currentPage - 1) * itemsPerPage;
    return sorted.slice(start, start + itemsPerPage);
  }, [sorted, currentPage]);

  const handleSort = (key: SortKey) => {
    if (key === sortKey) setSortAsc(!sortAsc);
    else { setSortKey(key); setSortAsc(key === 'target_name'); }
  };

  const headerClass = "px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase cursor-pointer hover:text-gray-700";

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">Unified results explorer</h2>

      <div className="flex flex-wrap gap-3 mb-4">
        <input
          type="text"
          placeholder="Search... (* = wildcard)"
          value={search}
          onChange={e => { setSearch(e.target.value); setCurrentPage(1); }}
          className="px-3 py-1.5 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 flex-1 min-w-[200px] max-w-md"
        />
        <select
          value={sourceFilter}
          onChange={e => { setSourceFilter(e.target.value as SourceFilter); setCurrentPage(1); }}
          className="px-3 py-1.5 border border-gray-300 rounded-md text-sm"
        >
          <option value="all">All sources</option>
          <option value="cal">Calibration (X*w)</option>
          <option value="val">Validation (Sim)</option>
        </select>
        <select
          value={geoFilter}
          onChange={e => { setGeoFilter(e.target.value as GeoFilter); setCurrentPage(1); }}
          className="px-3 py-1.5 border border-gray-300 rounded-md text-sm"
        >
          <option value="all">All geo levels</option>
          <option value="national">National</option>
          <option value="district">District</option>
          <option value="state">State</option>
        </select>
        <select
          value={variableFilter}
          onChange={e => { setVariableFilter(e.target.value); setCurrentPage(1); }}
          className="px-3 py-1.5 border border-gray-300 rounded-md text-sm"
        >
          <option value="">All variables</option>
          {variables.map(v => <option key={v} value={v}>{v}</option>)}
        </select>
      </div>

      <p className="text-xs text-gray-500 mb-2">{sorted.length} of {allRows.length} rows</p>

      <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50 sticky top-0">
            <tr>
              <th className={headerClass} onClick={() => handleSort('source')}>Source</th>
              <th className={headerClass} onClick={() => handleSort('geo')}>Geo</th>
              <th className={headerClass} onClick={() => handleSort('target_name')}>Target Name</th>
              <th className={headerClass} onClick={() => handleSort('target_value')}>Target</th>
              <th className={headerClass} onClick={() => handleSort('estimate')}>Estimate</th>
              <th className={headerClass} onClick={() => handleSort('rel_abs_error')}>Rel Abs Error</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {paginated.map((r, i) => {
              const errorColor = r.rel_abs_error < 0.05 ? 'text-green-600'
                : r.rel_abs_error < 0.20 ? 'text-yellow-600'
                : 'text-red-600';
              return (
                <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                  <td className="px-3 py-2 text-sm">
                    <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${
                      r.source === 'cal'
                        ? 'bg-blue-100 text-blue-700'
                        : 'bg-purple-100 text-purple-700'
                    }`}>
                      {r.source === 'cal' ? 'Cal' : 'Val'}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-sm text-gray-700">{r.geo || '—'}</td>
                  <td className="px-3 py-2 text-sm font-mono text-gray-900 max-w-md truncate" title={r.target_name}>
                    {r.target_name}
                  </td>
                  <td className="px-3 py-2 text-sm text-gray-700 text-right">{formatNumber(r.target_value)}</td>
                  <td className="px-3 py-2 text-sm text-gray-700 text-right">{formatNumber(r.estimate)}</td>
                  <td className="px-3 py-2 text-sm text-right">
                    <span className={`font-medium ${errorColor}`}>
                      {isFinite(r.rel_abs_error) ? `${(r.rel_abs_error * 100).toFixed(1)}%` : 'Inf'}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="flex justify-between items-center mt-4">
          <button
            onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
            disabled={currentPage === 1}
            className="px-4 py-2 bg-blue-600 text-white rounded disabled:bg-gray-300 disabled:text-gray-500 text-sm"
          >
            Previous
          </button>
          <span className="text-sm text-gray-600">Page {currentPage} of {totalPages}</span>
          <button
            onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
            disabled={currentPage === totalPages}
            className="px-4 py-2 bg-blue-600 text-white rounded disabled:bg-gray-300 disabled:text-gray-500 text-sm"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
