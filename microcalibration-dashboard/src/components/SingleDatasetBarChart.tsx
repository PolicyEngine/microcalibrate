'use client';

import { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { CalibrationDataPoint } from '@/types/calibration';
import { BarChart3 } from 'lucide-react';
import { sortTargetsWithRelevance, getSortedUniqueTargets } from '@/utils/targetOrdering';

interface SingleDatasetBarChartProps {
  data: CalibrationDataPoint[];
}

export default function SingleDatasetBarChart({ data }: SingleDatasetBarChartProps) {
  const [selectedEpoch, setSelectedEpoch] = useState<number | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [currentPage, setCurrentPage] = useState<number>(0);
  const [showTargetLabels, setShowTargetLabels] = useState<boolean>(false);
  const MAX_DISPLAYED_TARGETS = 15;

  // Get all unique targets and sort them
  const allTargets = getSortedUniqueTargets(data);

  // Get all available epochs
  const availableEpochs = Array.from(new Set(data.map(d => d.epoch))).sort((a, b) => b - a);

  // Initialize selected epoch to the latest
  useEffect(() => {
    if (availableEpochs.length > 0 && selectedEpoch === null) {
      setSelectedEpoch(availableEpochs[0]);
    }
  }, [availableEpochs, selectedEpoch]);

  // Reset page when search changes
  useEffect(() => {
    setCurrentPage(0);
  }, [searchQuery]);

  if (data.length === 0) {
    return (
      <div className="bg-white border border-gray-300 p-6 rounded-lg shadow-sm">
        <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <BarChart3 className="w-5 h-5 mr-2" />
          Target estimates comparison
        </h2>
        <div className="bg-yellow-50 border border-yellow-200 rounded p-4">
          <p className="text-yellow-800">No data available for bar chart visualization.</p>
        </div>
      </div>
    );
  }

  // Filter and paginate targets based on search
  const getFilteredTargets = () => {
    const filtered = allTargets.filter(target => 
      target.toLowerCase().includes(searchQuery.toLowerCase())
    );
    return sortTargetsWithRelevance(filtered, searchQuery);
  };

  const filteredTargets = getFilteredTargets();
  const totalPages = Math.ceil(filteredTargets.length / MAX_DISPLAYED_TARGETS);
  const startIndex = currentPage * MAX_DISPLAYED_TARGETS;
  const endIndex = startIndex + MAX_DISPLAYED_TARGETS;
  const currentPageTargets = filteredTargets.slice(startIndex, endIndex);

  // Prepare bar chart data
  const prepareBarChartData = () => {
    if (selectedEpoch === null) return [];

    const barData: Array<{
      targetName: string;
      target: number;
      estimate: number;
      relativeError: number;
      displayName: string;
    }> = [];

    currentPageTargets.forEach((targetName, index) => {
      const point = data.find(d => d.target_name === targetName && d.epoch === selectedEpoch);
      
      if (point) {
        const displayName = showTargetLabels ? targetName : `T${startIndex + index + 1}`;
        barData.push({
          targetName,
          target: point.target,
          estimate: point.estimate,
          relativeError: point.rel_abs_error * 100, // Convert to percentage
          displayName
        });
      }
    });

    return barData;
  };

  const barChartData = prepareBarChartData();

  const formatValueCompact = (value: number) => {
    if (value === 0) return '0';
    const abs = Math.abs(value);
    
    if (abs >= 1e6) {
      return (value / 1e6).toPrecision(2) + 'M';
    } else if (abs >= 1e3) {
      return (value / 1e3).toPrecision(2) + 'K';
    } else if (abs >= 1) {
      return value.toPrecision(2);
    } else {
      return value.toPrecision(2);
    }
  };


  return (
    <div className="bg-white border border-gray-300 p-6 rounded-lg shadow-sm">
      <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
        <BarChart3 className="w-5 h-5 mr-2" />
        Target vs. estimates comparison
      </h2>

      {/* Controls Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-4">
            {/* Epoch Selector */}
            <div className="flex items-center space-x-2">
              <label htmlFor="epoch-select" className="text-sm font-medium text-gray-700">
                Epoch:
              </label>
              <select
                id="epoch-select"
                value={selectedEpoch || ''}
                onChange={(e) => setSelectedEpoch(Number(e.target.value))}
                className="border border-gray-300 rounded-md px-2 py-1 text-sm text-gray-900 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {availableEpochs.map(epoch => (
                  <option key={epoch} value={epoch}>
                    {epoch}
                  </option>
                ))}
              </select>
            </div>

            {/* Target Labels Toggle */}
            <div className="flex items-center space-x-2">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={showTargetLabels}
                  onChange={(e) => setShowTargetLabels(e.target.checked)}
                  className="mr-2 rounded"
                />
                <span className="text-xs text-gray-700">Show target labels</span>
              </label>
            </div>
          </div>
        </div>

        {/* Target Search and Pagination */}
        <div className="p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-semibold text-gray-800">Search targets to display</h4>
            <div className="flex items-center space-x-2">
              <span className="text-xs text-gray-600">
                {filteredTargets.length} matches (showing {currentPageTargets.length})
              </span>
            </div>
          </div>
          
          {/* Search Bar */}
          <div className="mb-3">
            <input
              type="text"
              placeholder="Search targets by name..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>

          {/* Results Info */}
          {searchQuery && (
            <div className="mb-3 text-xs text-gray-600">
              Found {filteredTargets.length} target{filteredTargets.length !== 1 ? 's' : ''} containing &quot;{searchQuery}&quot;
              {filteredTargets.length > MAX_DISPLAYED_TARGETS && (
                <span className="ml-2 text-yellow-700">
                  (showing {MAX_DISPLAYED_TARGETS} per page)
                </span>
              )}
            </div>
          )}

          {/* Pagination Controls */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setCurrentPage(Math.max(0, currentPage - 1))}
                  disabled={currentPage === 0}
                  className="px-2 py-1 text-xs bg-blue-100 text-blue-700 rounded hover:bg-blue-200 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Previous
                </button>
                <span className="text-xs text-gray-600">
                  Page {currentPage + 1} of {totalPages}
                </span>
                <button
                  onClick={() => setCurrentPage(Math.min(totalPages - 1, currentPage + 1))}
                  disabled={currentPage === totalPages - 1}
                  className="px-2 py-1 text-xs bg-blue-100 text-blue-700 rounded hover:bg-blue-200 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Next
                </button>
              </div>
              <div className="text-xs text-gray-500">
                Showing {startIndex + 1}-{Math.min(endIndex, filteredTargets.length)} of {filteredTargets.length}
              </div>
            </div>
          )}

          {/* Current Page Targets Preview */}
          {currentPageTargets.length > 0 && (
            <div className="text-xs text-gray-600">
              <strong>Current page targets:</strong> {currentPageTargets.slice(0, 5).join(', ')}
              {currentPageTargets.length > 5 && ` and ${currentPageTargets.length - 5} more...`}
            </div>
          )}
        </div>
      </div>

      {/* Bar Chart */}
      {barChartData.length > 0 && selectedEpoch !== null ? (
        <div className="h-96">
          <ResponsiveContainer width="100%" height={450}>
            <BarChart
              data={barChartData}
              margin={{ top: 10, right: 30, left: 20, bottom: showTargetLabels ? 120 : 60 }}
              barGap={2}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="displayName" 
                angle={showTargetLabels ? -45 : 0}
                textAnchor={showTargetLabels ? "end" : "middle"}
                height={showTargetLabels ? 80 : 40}
                interval={0}
                tick={showTargetLabels ? { fontSize: 10 } : false}
                label={{ 
                  value: showTargetLabels ? '' : 'Target', 
                  position: 'insideBottom', 
                  fontSize: 12 
                }}
              />
              <YAxis
                tickFormatter={formatValueCompact}
                tick={{ fontSize: 10 }}
                label={{ value: 'Value', angle: -90, position: 'insideLeft', textAnchor: 'middle', fontSize: 12, dx: -10 }}
              />
              <Tooltip 
                formatter={(value: number, name: string) => {
                  const labels = {
                    target: 'Target value',
                    estimate: 'Estimate'
                  };
                  return [formatValueCompact(value), labels[name as keyof typeof labels] || name];
                }}
                labelFormatter={(label, payload) => {
                  if (payload && payload[0] && payload[0].payload) {
                    return `Target: ${payload[0].payload.targetName}`;
                  }
                  return `Target: ${label}`;
                }}
              />
              <Legend 
                verticalAlign="top"
                height={0}
                content={(props: unknown) => {
                  const payload = (props as {payload?: Array<{dataKey: string, value: number, color: string}>})?.payload;
                  const barItems = payload || [];
                  
                  return (
                    <div className="flex flex-col items-center space-y-2 pb-4">
                      {barItems.map((item, index: number) => (
                        <div
                          key={index}
                          className="flex items-center space-x-2"
                        >
                          <div
                            className="w-6 h-4 border"
                            style={{
                              backgroundColor: item.color,
                              borderColor: item.color
                            }}
                          />
                          <span className="font-medium" style={{ color: item.color,  fontSize: '11px'  }}>
                            {item.value}
                          </span>
                        </div>
                      ))}
                    </div>
                  );
                }}
              />
              
              {/* Target Value Bars */}
              <Bar dataKey="target" name="Target value" fill="#16a34a" />
              
              {/* Estimates */}
              <Bar dataKey="estimate" name="Estimate" fill="#2563eb" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="h-32 flex items-center justify-center bg-gray-50 rounded-lg">
          <p className="text-gray-600">
            {filteredTargets.length === 0 && searchQuery
              ? `No targets found matching "${searchQuery}"`
              : filteredTargets.length === 0
              ? 'Enter a search term to find targets'
              : selectedEpoch === null 
              ? 'Select an epoch to display comparison'
              : 'No data available for current targets and epoch'
            }
          </p>
        </div>
      )}

      {/* Summary info */}
      {barChartData.length > 0 && (
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 mb-2">Current view summary</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Targets shown:</span>
              <div className="font-medium">{barChartData.length}</div>
            </div>
            <div>
              <span className="text-gray-600">Avg rel error:</span>
              <div className="font-medium">
                {(barChartData.reduce((sum, d) => sum + d.relativeError, 0) / barChartData.length).toFixed(1)}%
              </div>
            </div>
            <div>
              <span className="text-gray-600">Best target:</span>
              <div className="font-medium text-green-600">
                {Math.min(...barChartData.map(d => d.relativeError)).toFixed(1)}%
              </div>
            </div>
            <div>
              <span className="text-gray-600">Worst target:</span>
              <div className="font-medium text-red-600">
                {Math.max(...barChartData.map(d => d.relativeError)).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}