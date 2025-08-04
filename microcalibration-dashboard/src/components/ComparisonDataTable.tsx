'use client';

import { CalibrationDataPoint } from '@/types/calibration';
import { useState, useMemo } from 'react';
import { compareTargetNames } from '@/utils/targetOrdering';

interface ComparisonDataTableProps {
  firstData: CalibrationDataPoint[];
  secondData: CalibrationDataPoint[];
  firstName: string;
  secondName: string;
}

interface ComparisonRow {
  targetName: string;
  epoch: number;
  first?: CalibrationDataPoint;
  second?: CalibrationDataPoint;
}

type SortField = keyof CalibrationDataPoint | 'random' | 'difference';
type SortDataset = 'first' | 'second' | null;
type SortDirection = 'asc' | 'desc';

export default function ComparisonDataTable({ 
  firstData, 
  secondData, 
  firstName, 
  secondName 
}: ComparisonDataTableProps) {
  // Get all unique epochs from both datasets
  const allEpochs = useMemo(() => {
    const epochs1 = firstData.map(item => item.epoch);
    const epochs2 = secondData.map(item => item.epoch);
    return Array.from(new Set([...epochs1, ...epochs2])).sort((a, b) => a - b);
  }, [firstData, secondData]);

  // Get the maximum epoch for default selection
  const maxEpoch = allEpochs.length > 0 ? Math.max(...allEpochs) : 0;
  
  const [sortField, setSortField] = useState<SortField>('target_name');
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc');
  const [sortDataset, setSortDataset] = useState<SortDataset>(null);
  const [filter, setFilter] = useState('');
  const [epochFilter, setEpochFilter] = useState(maxEpoch);
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 50;

  // Create comparison rows by merging data from both datasets
  const comparisonRows = useMemo(() => {
    // Get all unique target names from both datasets
    const targetNames1 = new Set(firstData.map(item => item.target_name));
    const targetNames2 = new Set(secondData.map(item => item.target_name));
    const allTargetNames = Array.from(new Set([...targetNames1, ...targetNames2]));

    // Create lookup maps for efficient data retrieval
    const firstDataMap = new Map<string, CalibrationDataPoint>();
    const secondDataMap = new Map<string, CalibrationDataPoint>();

    firstData.forEach(item => {
      if (item.epoch === epochFilter) {
        firstDataMap.set(item.target_name, item);
      }
    });

    secondData.forEach(item => {
      if (item.epoch === epochFilter) {
        secondDataMap.set(item.target_name, item);
      }
    });

    // Create comparison rows
    const rows: ComparisonRow[] = allTargetNames.map(targetName => ({
      targetName,
      epoch: epochFilter,
      first: firstDataMap.get(targetName),
      second: secondDataMap.get(targetName),
    }));

    // Filter by search term
    return rows.filter(row => 
      row.targetName.toLowerCase().includes(filter.toLowerCase())
    );
  }, [firstData, secondData, epochFilter, filter]);

  const sortedData = useMemo(() => {
    if (sortField === 'random') {
      return [...comparisonRows].sort(() => {
        const seed = comparisonRows.length;
        return (seed * 9301 + 49297) % 233280 / 233280 - 0.5;
      });
    }
    
    return [...comparisonRows].sort((a, b) => {
      if (sortField === 'target_name') {
        const result = compareTargetNames(a.targetName, b.targetName);
        return sortDirection === 'asc' ? result : -result;
      }

      if (sortField === 'difference') {
        // Calculate differences for sorting
        const aDiff = (a.first?.rel_abs_error !== undefined && a.second?.rel_abs_error !== undefined) 
          ? a.second.rel_abs_error - a.first.rel_abs_error 
          : null;
        const bDiff = (b.first?.rel_abs_error !== undefined && b.second?.rel_abs_error !== undefined) 
          ? b.second.rel_abs_error - b.first.rel_abs_error 
          : null;
        
        // Handle undefined/null values - put them at the end regardless of sort direction
        if (aDiff === null && bDiff === null) {
          return 0;
        }
        if (aDiff === null) {
          return 1;
        }
        if (bDiff === null) {
          return -1;
        }
        
        return sortDirection === 'asc' ? aDiff - bDiff : bDiff - aDiff;
      }

      // Sort by values based on selected dataset
      let aVal, bVal;
      if (sortDataset === 'first') {
        aVal = a.first?.[sortField as keyof CalibrationDataPoint];
        bVal = b.first?.[sortField as keyof CalibrationDataPoint];
      } else if (sortDataset === 'second') {
        aVal = a.second?.[sortField as keyof CalibrationDataPoint];
        bVal = b.second?.[sortField as keyof CalibrationDataPoint];
      } else {
        // Default behavior: use first dataset, fallback to second
        aVal = a.first?.[sortField as keyof CalibrationDataPoint] ?? a.second?.[sortField as keyof CalibrationDataPoint];
        bVal = b.first?.[sortField as keyof CalibrationDataPoint] ?? b.second?.[sortField as keyof CalibrationDataPoint];
      }
      
      // Handle undefined/null values - put them at the end regardless of sort direction
      if ((aVal === undefined || aVal === null) && (bVal === undefined || bVal === null)) {
        return 0;
      }
      if (aVal === undefined || aVal === null) {
        return 1;
      }
      if (bVal === undefined || bVal === null) {
        return -1;
      }
      
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
      }
      
      const aStr = String(aVal || '').toLowerCase();
      const bStr = String(bVal || '').toLowerCase();
      return sortDirection === 'asc' 
        ? aStr.localeCompare(bStr)
        : bStr.localeCompare(aStr);
    });
  }, [comparisonRows, sortField, sortDirection, sortDataset]);

  const paginatedData = useMemo(() => {
    const start = (currentPage - 1) * itemsPerPage;
    return sortedData.slice(start, start + itemsPerPage);
  }, [sortedData, currentPage]);
  
  if (firstData.length === 0 && secondData.length === 0) {
    return (
      <div className="bg-white border border-gray-300 p-6 rounded-lg shadow-sm">
        <h2 className="text-xl font-bold text-gray-800">Detailed comparison results</h2>
        <p className="text-gray-600 mt-4">No data available</p>
      </div>
    );
  }

  const totalPages = Math.ceil(sortedData.length / itemsPerPage);

  const handleSort = (field: keyof CalibrationDataPoint | 'difference', dataset?: 'first' | 'second') => {
    // For target_name and difference, don't use dataset-specific sorting
    if (field === 'target_name' || field === 'difference') {
      if (sortField === field && sortDataset === null) {
        setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
      } else {
        setSortField(field);
        setSortDataset(null);
        setSortDirection(field === 'difference' ? 'desc' : 'asc'); // Start with desc for difference to show biggest improvements first
      }
      return;
    }

    // For other fields, use dataset-specific sorting
    // Ensure dataset is always defined for non-target_name fields
    const targetDataset = dataset || 'first';
    if (sortField === field && sortDataset === targetDataset) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDataset(targetDataset);
      setSortDirection('desc');
    }
  };

  const SortButton = ({ field, children, dataset }: { 
    field: keyof CalibrationDataPoint | 'difference', 
    children: React.ReactNode, 
    dataset?: 'first' | 'second'
  }) => {
    const isActive = (field === 'target_name' || field === 'difference') 
      ? (sortField === field && sortDataset === null)
      : (sortField === field && sortDataset === dataset);
    
    return (
      <div className="flex items-center gap-1">
        <span>{children}</span>
        <button
          onClick={() => handleSort(field, dataset)}
          className={`flex flex-col items-center justify-center w-4 h-6 transition-colors ${
            isActive ? 'text-blue-600' : 'text-gray-400 hover:text-gray-600'
          }`}
        >
          <span className={`text-xs leading-none ${
            isActive && sortDirection === 'asc' ? 'text-blue-600' : 'text-gray-300'
          }`}>▲</span>
          <span className={`text-xs leading-none ${
            isActive && sortDirection === 'desc' ? 'text-blue-600' : 'text-gray-300'
          }`}>▼</span>
        </button>
      </div>
    );
  };

  const DatasetSortButton = ({ field, dataset, children }: {
    field: keyof CalibrationDataPoint,
    dataset: 'first' | 'second',
    children: React.ReactNode
  }) => {
    const isActive = sortField === field && sortDataset === dataset;
    const isFirst = dataset === 'first';
    const colorClass = isFirst ? 'text-blue-600' : 'text-purple-600';
    
    return (
      <div className="flex items-center justify-center gap-1">
        <span className={`text-xs font-medium ${colorClass}`}>{children}</span>
        <button
          onClick={() => handleSort(field, dataset)}
          className={`flex flex-col items-center justify-center w-4 h-6 transition-colors ${
            isActive ? colorClass : 'text-gray-400 hover:text-gray-600'
          }`}
          title={`Sort by ${field} for ${isFirst ? 'first' : 'second'} dataset`}
        >
          <span className={`text-xs leading-none ${
            isActive && sortDirection === 'asc' ? colorClass : 'text-gray-300'
          }`}>▲</span>
          <span className={`text-xs leading-none ${
            isActive && sortDirection === 'desc' ? colorClass : 'text-gray-300'
          }`}>▼</span>
        </button>
      </div>
    );
  };

  const formatValue = (value: number | undefined | null) => {
    if (value === undefined || value === null || isNaN(value)) {
      return 'N/A';
    }
    
    const numValue = Number(value);
    if (isNaN(numValue)) {
      return 'N/A';
    }
    
    if (Math.abs(numValue) >= 1000000) {
      return (numValue / 1000000).toFixed(2) + 'M';
    } else if (Math.abs(numValue) >= 1000) {
      return (numValue / 1000).toFixed(1) + 'K';
    }
    return numValue.toFixed(2);
  };

  const getErrorClass = (relError: number | undefined | null) => {
    if (relError === undefined || relError === null || isNaN(relError)) {
      return 'text-gray-400';
    }
    return relError < 0.05 ? 'text-green-600' : 
           relError < 0.20 ? 'text-yellow-600' : 'text-red-600';
  };

  const renderDataCell = (value: number | undefined | null, isFirst: boolean, isError: boolean = false) => {
    if (value === undefined || value === null) {
      return <span className="text-gray-400 font-mono text-xs">—</span>;
    }

    const colorClass = isFirst ? 'text-blue-600' : 'text-purple-600';
    let finalClass = colorClass;
    let displayValue = formatValue(value);
    
    if (isError && typeof value === 'number' && !isNaN(value)) {
      displayValue = `${(value * 100).toFixed(2)}%`;
      finalClass = getErrorClass(value);
    } else if (typeof value === 'number' && value >= 0 && !isError) {
      displayValue = '+' + displayValue;
    }

    return (
      <span className={`font-mono text-xs font-semibold ${finalClass}`}>
        {displayValue}
      </span>
    );
  };

  const renderDifferenceCell = (firstValue: number | undefined | null, secondValue: number | undefined | null) => {
    if (firstValue === undefined || firstValue === null || secondValue === undefined || secondValue === null) {
      return <span className="text-gray-400 font-mono text-xs">—</span>;
    }

    const difference = secondValue - firstValue;
    const percentageDiff = (difference * 100).toFixed(2);
    const sign = difference > 0 ? '+' : '';
    
    // Color coding: negative difference (B better than A) = green, positive (B worse than A) = red
    let colorClass = 'text-gray-600';
    if (Math.abs(difference) > 0.001) { // Only color if difference is meaningful
      colorClass = difference < 0 ? 'text-green-600' : 'text-red-600';
    }

    return (
      <span className={`font-mono text-xs font-semibold ${colorClass}`} title={`B: ${(secondValue * 100).toFixed(2)}% - A: ${(firstValue * 100).toFixed(2)}% = ${sign}${percentageDiff}%`}>
        {sign}{percentageDiff}%
      </span>
    );
  };

  return (
    <div className="bg-white border border-gray-300 p-6 rounded-lg shadow-sm">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold text-gray-800">Detailed comparison results</h2>
        <div className="flex space-x-4">
          <select
            value={epochFilter}
            onChange={(e) => {
              setEpochFilter(parseInt(e.target.value));
              setCurrentPage(1);
            }}
            className="bg-white border border-gray-300 text-gray-900 px-3 py-2 rounded font-mono text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {allEpochs.sort((a, b) => b - a).map(epoch => (
              <option key={epoch} value={epoch}>Epoch {epoch}</option>
            ))}
          </select>
          <input
            type="text"
            placeholder="Search target names..."
            value={filter}
            onChange={(e) => {
              setFilter(e.target.value);
              setCurrentPage(1);
            }}
            className="bg-white border border-gray-300 text-gray-900 px-3 py-2 rounded font-mono text-sm w-64 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <div className="text-gray-600 py-2 text-sm">
            {sortedData.length} entries
          </div>
        </div>
      </div>

      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-blue-600 rounded"></div>
            <span className="text-sm text-gray-700 font-medium">{firstName}</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-purple-600 rounded"></div>
            <span className="text-sm text-gray-700 font-medium">{secondName}</span>
          </div>
        </div>
        <div className="text-xs text-gray-500 space-y-1">
          <div>💡 Click the ▲▼ arrows next to column headers to sort. Target name sorts alphabetically, A/B arrows under &quot;Rel abs error %&quot; sort by dataset performance.</div>
          <div>* indicates targets that exist in only one dataset (hover for details)</div>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm font-mono table-fixed">
          <colgroup>
            <col className="w-40" />
            <col className="w-20" />
            <col className="w-20" />
            <col className="w-20" />
            <col className="w-20" />
            <col className="w-20" />
            <col className="w-20" />
            <col className="w-20" />
            <col className="w-20" />
            <col className="w-20" />
            <col className="w-20" />
            <col className="w-24" />
          </colgroup>
          <thead>
            <tr className="border-b border-gray-300 bg-gray-50">
              <th className="text-left py-3 px-4 font-semibold text-gray-700" rowSpan={2}>
                <SortButton field="target_name">Target name</SortButton>
              </th>
              <th className="text-center py-2 px-4 font-semibold text-gray-700 border-b border-gray-200" colSpan={2}>
                Target value
              </th>
              <th className="text-center py-2 px-4 font-semibold text-gray-700 border-b border-gray-200" colSpan={2}>
                Estimate
              </th>
              <th className="text-center py-2 px-4 font-semibold text-gray-700 border-b border-gray-200" colSpan={2}>
                Error
              </th>
              <th className="text-center py-2 px-4 font-semibold text-gray-700 border-b border-gray-200" colSpan={2}>
                Abs error
              </th>
              <th className="text-center py-2 px-4 font-semibold text-gray-700 border-b border-gray-200" colSpan={2}>
                Rel abs error %
              </th>
              <th className="text-center py-2 px-4 font-semibold text-gray-700 border-b border-gray-200">
                Difference
              </th>
            </tr>
            <tr className="border-b border-gray-300 bg-gray-50">
              <th className="text-center py-2 px-2 text-xs font-medium text-blue-600">A</th>
              <th className="text-center py-2 px-2 text-xs font-medium text-purple-600">B</th>
              <th className="text-center py-2 px-2 text-xs font-medium text-blue-600">A</th>
              <th className="text-center py-2 px-2 text-xs font-medium text-purple-600">B</th>
              <th className="text-center py-2 px-2 text-xs font-medium text-blue-600">A</th>
              <th className="text-center py-2 px-2 text-xs font-medium text-purple-600">B</th>
              <th className="text-center py-2 px-2 text-xs font-medium text-blue-600">A</th>
              <th className="text-center py-2 px-2 text-xs font-medium text-purple-600">B</th>
              <th className="text-center py-2 px-2">
                <DatasetSortButton field="rel_abs_error" dataset="first">
                  A
                </DatasetSortButton>
              </th>
              <th className="text-center py-2 px-2">
                <DatasetSortButton field="rel_abs_error" dataset="second">
                  B
                </DatasetSortButton>
              </th>
              <th className="text-center py-2 px-5">
                <SortButton field="difference">(B-A)</SortButton>
              </th>
            </tr>
          </thead>
          <tbody>
            {paginatedData.map((row, i) => (
              <tr key={`${row.targetName}-${row.epoch}-${i}`} className="border-b border-gray-200 hover:bg-gray-50">
                <td className="py-3 px-4 text-gray-900" title={row.targetName}>
                  <div 
                    className="overflow-x-auto whitespace-nowrap [&::-webkit-scrollbar]:hidden flex items-center gap-1" 
                    style={{ 
                      scrollbarWidth: 'none', 
                      msOverflowStyle: 'none',
                    }}
                  >
                    <span>{row.targetName}</span>
                    {!row.first && row.second && (
                      <span 
                        className="text-purple-500 text-xs cursor-help" 
                        title={`Only in ${secondName}`}
                      >
                        *
                      </span>
                    )}
                    {row.first && !row.second && (
                      <span 
                        className="text-blue-500 text-xs cursor-help" 
                        title={`Only in ${firstName}`}
                      >
                        *
                      </span>
                    )}
                  </div>
                </td>
                
                {/* Target value columns */}
                <td className="py-3 px-2 text-right">
                  {renderDataCell(row.first?.target, true)}
                </td>
                <td className="py-3 px-2 text-right">
                  {renderDataCell(row.second?.target, false)}
                </td>
                
                {/* Estimate columns */}
                <td className="py-3 px-2 text-right">
                  {renderDataCell(row.first?.estimate, true)}
                </td>
                <td className="py-3 px-2 text-right">
                  {renderDataCell(row.second?.estimate, false)}
                </td>
                
                {/* Error columns */}
                <td className="py-3 px-2 text-right">
                  {renderDataCell(row.first?.error, true)}
                </td>
                <td className="py-3 px-2 text-right">
                  {renderDataCell(row.second?.error, false)}
                </td>
                
                {/* Abs error columns */}
                <td className="py-3 px-2 text-right">
                  {renderDataCell(row.first?.abs_error, true)}
                </td>
                <td className="py-3 px-2 text-right">
                  {renderDataCell(row.second?.abs_error, false)}
                </td>
                
                {/* Rel abs error columns */}
                <td className="py-3 px-2 text-right">
                  {renderDataCell(row.first?.rel_abs_error, true, true)}
                </td>
                <td className="py-3 px-2 text-right">
                  {renderDataCell(row.second?.rel_abs_error, false, true)}
                </td>
                
                {/* Difference column */}
                <td className="py-3 px-2 text-center">
                  {renderDifferenceCell(row.first?.rel_abs_error, row.second?.rel_abs_error)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="flex justify-between items-center mt-6">
          <button
            onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
            disabled={currentPage === 1}
            className="px-4 py-2 bg-blue-600 text-white rounded disabled:bg-gray-300 disabled:text-gray-500 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            Previous
          </button>
          <div className="text-gray-600 font-mono text-sm">
            Page {currentPage} of {totalPages}
          </div>
          <button
            onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
            disabled={currentPage === totalPages}
            className="px-4 py-2 bg-blue-600 text-white rounded disabled:bg-gray-300 disabled:text-gray-500 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}