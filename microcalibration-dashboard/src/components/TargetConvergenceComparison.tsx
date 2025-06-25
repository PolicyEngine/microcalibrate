'use client';

import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ReferenceLine, BarChart, Bar } from 'recharts';
import { CalibrationDataPoint } from '@/types/calibration';
import { Target } from 'lucide-react';

interface TargetConvergenceComparisonProps {
  firstData: CalibrationDataPoint[];
  secondData: CalibrationDataPoint[];
  firstName: string;
  secondName: string;
}

interface ConvergenceDataPoint {
  epoch: number;
  target: number;
  firstEstimate: number | null;
  secondEstimate: number | null;
  firstError: number | null;
  secondError: number | null;
}

export default function TargetConvergenceComparison({ 
  firstData, 
  secondData, 
  firstName, 
  secondName 
}: TargetConvergenceComparisonProps) {
  // Find overlapping targets
  const firstTargets = new Set(firstData.map(d => d.target_name));
  const secondTargets = new Set(secondData.map(d => d.target_name));
  const commonTargets = Array.from(firstTargets).filter(target => secondTargets.has(target)).sort();

  const [selectedTarget, setSelectedTarget] = useState<string>(commonTargets[0] || '');
  const [lineOpacity, setLineOpacity] = useState({
    firstEstimate: 1,
    secondEstimate: 1
  });

  // Bar chart states
  const [selectedEpoch, setSelectedEpoch] = useState<number | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [currentPage, setCurrentPage] = useState<number>(0);
  const MAX_DISPLAYED_TARGETS = 15;

  if (commonTargets.length === 0) {
    return (
      <div className="bg-white border border-gray-300 p-6 rounded-lg shadow-sm">
        <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <Target className="w-5 h-5 mr-2" />
          Target convergence comparison
        </h2>
        <div className="bg-yellow-50 border border-yellow-200 rounded p-4">
          <p className="text-yellow-800">
            No common targets found between datasets for convergence comparison.
          </p>
        </div>
      </div>
    );
  }

  // Prepare data for the selected target
  const prepareConvergenceData = (targetName: string): ConvergenceDataPoint[] => {
    const firstTargetData = firstData.filter(d => d.target_name === targetName).sort((a, b) => a.epoch - b.epoch);
    const secondTargetData = secondData.filter(d => d.target_name === targetName).sort((a, b) => a.epoch - b.epoch);

    // Get all epochs from both datasets
    const allEpochs = new Set([
      ...firstTargetData.map(d => d.epoch),
      ...secondTargetData.map(d => d.epoch)
    ]);

    const convergenceData: ConvergenceDataPoint[] = [];

    Array.from(allEpochs).sort((a, b) => a - b).forEach(epoch => {
      const firstPoint = firstTargetData.find(d => d.epoch === epoch);
      const secondPoint = secondTargetData.find(d => d.epoch === epoch);

      // Use the target value from whichever dataset has data for this epoch
      const targetValue = firstPoint?.target ?? secondPoint?.target ?? 0;

      convergenceData.push({
        epoch,
        target: targetValue,
        firstEstimate: firstPoint?.estimate ?? null,
        secondEstimate: secondPoint?.estimate ?? null,
        firstError: firstPoint?.rel_abs_error ?? null,
        secondError: secondPoint?.rel_abs_error ?? null
      });
    });

    return convergenceData;
  };

  const convergenceData = prepareConvergenceData(selectedTarget);
  const targetValue = convergenceData[0]?.target ?? 0;

  // Initialize bar chart selections
  useEffect(() => {
    // Get all available epochs from both datasets
    const allEpochs = new Set([
      ...firstData.map(d => d.epoch),
      ...secondData.map(d => d.epoch)
    ]);
    const sortedEpochs = Array.from(allEpochs).sort((a, b) => b - a);
    
    // Set default epoch to the latest
    if (sortedEpochs.length > 0 && selectedEpoch === null) {
      setSelectedEpoch(sortedEpochs[0]);
    }

    // No need to set default targets anymore - handled by search/pagination
  }, [firstData, secondData, commonTargets, selectedEpoch]);

  // Filter and paginate targets based on search
  const getFilteredTargets = () => {
    const filtered = commonTargets.filter(target => 
      target.toLowerCase().includes(searchQuery.toLowerCase())
    );
    return filtered;
  };

  const filteredTargets = getFilteredTargets();
  const totalPages = Math.ceil(filteredTargets.length / MAX_DISPLAYED_TARGETS);
  const startIndex = currentPage * MAX_DISPLAYED_TARGETS;
  const endIndex = startIndex + MAX_DISPLAYED_TARGETS;
  const currentPageTargets = filteredTargets.slice(startIndex, endIndex);

  // Reset page when search changes
  useEffect(() => {
    setCurrentPage(0);
  }, [searchQuery]);

  // Prepare bar chart data
  const prepareBarChartData = () => {
    if (selectedEpoch === null) return [];

    const barData: Array<{
      targetName: string;
      targetValue: number;
      firstEstimate: number | null;
      secondEstimate: number | null;
    }> = [];

    currentPageTargets.forEach(targetName => {
      const firstPoint = firstData.find(d => d.target_name === targetName && d.epoch === selectedEpoch);
      const secondPoint = secondData.find(d => d.target_name === targetName && d.epoch === selectedEpoch);
      
      if (firstPoint || secondPoint) {
        barData.push({
          targetName,
          targetValue: firstPoint?.target ?? secondPoint?.target ?? 0,
          firstEstimate: firstPoint?.estimate ?? null,
          secondEstimate: secondPoint?.estimate ?? null
        });
      }
    });

    return barData;
  };

  const barChartData = prepareBarChartData();

  // Get available epochs
  const availableEpochs = Array.from(new Set([
    ...firstData.map(d => d.epoch),
    ...secondData.map(d => d.epoch)
  ])).sort((a, b) => b - a);

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

  // Calculate final statistics
  const finalFirstPoint = convergenceData.filter(d => d.firstEstimate !== null).slice(-1)[0];
  const finalSecondPoint = convergenceData.filter(d => d.secondEstimate !== null).slice(-1)[0];

  const formatValue = (value: number, precise = false) => {
    if (Math.abs(value) >= 1000000) {
      return (value / 1000000).toFixed(precise ? 0 : 0) + 'M';
    } else if (Math.abs(value) >= 1000) {
      return (value / 1000).toFixed(precise ? 2 : 1) + 'K';
    } else if (Math.abs(value) >= 1) {
      return value.toFixed(precise ? 3 : 0);
    } else {
      return value.toFixed(precise ? 5 : 3);
    }
  };

  const formatTooltip = (value: any, name: string) => {
    if (name === 'target') {
      return [formatValue(value), 'Target'];
    } else if (name === 'firstEstimate') {
      return [formatValue(value), `${firstName} estimate`];
    } else if (name === 'secondEstimate') {
      return [formatValue(value), `${secondName} estimate`];
    }
    return [formatValue(value), name];
  };

  const handleLegendClick = (data: any) => {
    const { dataKey } = data;
    if (dataKey === 'firstEstimate' || dataKey === 'secondEstimate') {
      setLineOpacity(prev => ({
        ...prev,
        [dataKey]: prev[dataKey as keyof typeof prev] === 1 ? 0.2 : 1
      }));
    }
  };

  const renderCustomLegend = ({ payload }: any) => {
    // put them in the order we want to show them
    const order = ['target', 'firstEstimate', 'secondEstimate'];
    const items = order
      .map(key => payload.find((p: any) => p.dataKey === key))
      .filter(Boolean);

    return (
      <div className="flex flex-col items-center space-y-2 pt-5">
        {items.map((item: any) => {
          const isFirst = item.dataKey === 'firstEstimate';
          const isSecond = item.dataKey === 'secondEstimate';

          return (
            <div
              key={item.dataKey}
              className="flex items-center space-x-2 cursor-pointer"
              onClick={() => {
                if (isFirst || isSecond) handleLegendClick(item);
              }}
            >
              {/* little line ▸ dashed for target, solid for others */}
              <div
                className="w-6 border-t-2"
                style={{
                  borderStyle: item.dataKey === 'target' ? 'dashed' : 'solid',
                  borderColor:
                    item.dataKey === 'target' ? '#dc2626' : item.color,
                }}
              />
              <span
                className="text-sm font-medium"
                style={{
                  color:
                    item.dataKey === 'target' ? '#dc2626' : item.color,
                  opacity:
                    isFirst
                      ? lineOpacity.firstEstimate
                      : isSecond
                      ? lineOpacity.secondEstimate
                      : 1,
                }}
              >
                {item.value}
              </span>
            </div>
          );
        })}
      </div>
    );
  };

  // Calculate Y-axis domain to avoid empty space
  const allValues = convergenceData.flatMap(d => [
    d.target,
    d.firstEstimate,
    d.secondEstimate
  ].filter(v => v !== null) as number[]);
  
  const minValue = Math.min(...allValues);
  const maxValue = Math.max(...allValues);
  const range = maxValue - minValue;
  const padding = range * 0.1; // 10% padding
  const yAxisDomain = [
    Math.max(0, minValue - padding),
    maxValue + padding
  ];

  return (
    <div className="bg-white border border-gray-300 p-6 rounded-lg shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold text-gray-800 flex items-center">
          <Target className="w-5 h-5 mr-2" />
          Target convergence comparison
        </h2>
        <div className="flex items-center space-x-3" style={{ maxWidth: '50%' }}>
          <label htmlFor="target-select" className="text-sm font-medium text-gray-700 whitespace-nowrap">
            Select target:
          </label>
          <div className="relative flex-1 min-w-0">
            <select
              id="target-select"
              value={selectedTarget}
              onChange={(e) => setSelectedTarget(e.target.value)}
              className="w-full border border-gray-300 rounded-md px-3 py-1 text-sm text-gray-900 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 overflow-x-auto"
              title={selectedTarget}
              style={{
                textOverflow: 'clip',
                whiteSpace: 'nowrap',
                scrollbarWidth: 'none',
                msOverflowStyle: 'none',
                WebkitOverflowScrolling: 'touch'
              }}
            >
              {commonTargets.map(target => (
                <option key={target} value={target} title={target}>
                  {target}
                </option>
              ))}
            </select>
            <style jsx>{`
              #target-select::-webkit-scrollbar {
                display: none;
              }
              #target-select {
                scrollbar-width: none;
                -ms-overflow-style: none;
              }
            `}</style>
          </div>
        </div>
      </div>

      {/* Summary statistics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6 p-4 bg-gray-50 rounded-lg">
        <div className="text-center">
          <div className="text-lg font-bold text-gray-900">{formatValue(targetValue)}</div>
          <div className="text-sm text-gray-600">Target value</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-blue-700">
            {finalFirstPoint?.firstEstimate ? formatValue(finalFirstPoint.firstEstimate) : 'N/A'}
          </div>
          <div className="text-sm text-gray-600">Final estimate (1st)</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-purple-700">
            {finalSecondPoint?.secondEstimate ? formatValue(finalSecondPoint.secondEstimate) : 'N/A'}
          </div>
          <div className="text-sm text-gray-600">Final estimate (2nd)</div>
        </div>
        <div className="text-center">
          <div className="flex flex-col space-y-1">
            <div className="text-sm font-bold text-blue-700">
              {finalFirstPoint?.firstError ? `${(finalFirstPoint.firstError * 100).toFixed(2)}%` : 'N/A'}
            </div>
            <div className="text-sm font-bold text-purple-700">
              {finalSecondPoint?.secondError ? `${(finalSecondPoint.secondError * 100).toFixed(2)}%` : 'N/A'}
            </div>
          </div>
          <div className="text-sm text-gray-600">Final errors</div>
        </div>
      </div>

      {/* Convergence Chart */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-3">Estimate convergence over epochs</h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={convergenceData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="epoch" 
                label={{ value: 'Epoch', position: 'insideBottom', offset: -10, fontSize: 12 }}
                tick={{ fontSize: 10 }}
              />
              <YAxis
                domain={yAxisDomain}
                tickFormatter={(value) => formatValue(value, true)}
                tickCount={8}
                tickMargin={14}
                label={{ value: 'Value', angle: -90, position: 'insideLeft', textAnchor: 'middle', dx: -19, fontSize: 12 }}
                tick={{ fontSize: 10 }}
              />
              <Tooltip 
                formatter={formatTooltip}
                labelFormatter={(label) => `Epoch: ${label}`}
              />
              <Legend 
                content={renderCustomLegend}
              />
              
              {/* Target reference line */}
              <ReferenceLine 
                y={targetValue} 
                stroke="#dc2626" 
                strokeWidth={3}
                strokeDasharray="5 5"
              />
              
              {/* First dataset estimates */}
              <Line 
                type="monotone" 
                dataKey="firstEstimate" 
                stroke="#3b82f6" 
                strokeWidth={2}
                strokeOpacity={lineOpacity.firstEstimate}
                dot={{ r: 3, fill: '#3b82f6', fillOpacity: lineOpacity.firstEstimate }}
                activeDot={{ r: 5, fillOpacity: lineOpacity.firstEstimate }}
                connectNulls={false}
                name={`${firstName}`}
              />
              
              {/* Second dataset estimates */}
              <Line 
                type="monotone" 
                dataKey="secondEstimate" 
                stroke="#7c3aed" 
                strokeWidth={2}
                strokeOpacity={lineOpacity.secondEstimate}
                dot={{ r: 3, fill: '#7c3aed', fillOpacity: lineOpacity.secondEstimate }}
                activeDot={{ r: 5, fillOpacity: lineOpacity.secondEstimate }}
                connectNulls={false}
                name={`${secondName}`}
              />
              
              {/* Invisible line for Target legend entry */}
              <Line 
                type="monotone" 
                dataKey="target" 
                stroke="#dc2626" 
                strokeWidth={3}
                strokeDasharray="5 5"
                dot={false}
                activeDot={false}
                connectNulls={false}
                name="Target"
                hide={true}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Relative Error Chart */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-3">Relative absolute error over epochs</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={convergenceData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="epoch" 
                label={{ value: 'Epoch', position: 'insideBottom', offset: -10, fontSize: 12 }}
                tick={{ fontSize: 10 }}
              />
              <YAxis
                scale="log"
                domain={['auto', 'auto']}
                tickFormatter={(value) => value < 0.001 ? value.toExponential(1) : value.toPrecision(3)}
                tickCount={6}
                tickMargin={14}
                label={{ value: 'Rel abs error (log)', angle: -90, position: 'insideLeft', textAnchor: 'middle', dx: -19, dy: 80, fontSize: 12 }}
                tick={{ fontSize: 10 }}
              />
              <Tooltip 
                formatter={(value: any, name: string) => {
                  if (name === 'firstError') {
                    return [`${(value * 100).toFixed(3)}%`, `${firstName} error`];
                  } else if (name === 'secondError') {
                    return [`${(value * 100).toFixed(3)}%`, `${secondName} error`];
                  }
                  return [value, name];
                }}
                labelFormatter={(label) => `Epoch: ${label}`}
              />
              <Legend 
                content={({ payload }: any) => {
                  const errorItems = payload?.filter((item: any) => 
                    item.dataKey === 'firstError' || item.dataKey === 'secondError'
                  ) || [];

                  return (
                    <div className="flex flex-col items-center space-y-2 pt-5">
                      {errorItems.map((item: any) => {
                        const isFirst = item.dataKey === 'firstError';
                        const currentOpacity = isFirst ? lineOpacity.firstEstimate : lineOpacity.secondEstimate;
                        
                        return (
                          <div
                            key={item.dataKey}
                            className="flex items-center space-x-2 cursor-pointer"
                            onClick={() => {
                              const targetKey = isFirst ? 'firstEstimate' : 'secondEstimate';
                              setLineOpacity(prev => ({
                                ...prev,
                                [targetKey]: prev[targetKey] === 1 ? 0.2 : 1
                              }));
                            }}
                          >
                            <div
                              className="w-6 border-t-2"
                              style={{
                                borderStyle: 'solid',
                                borderColor: item.color,
                              }}
                            />
                            <span
                              className="text-sm font-medium"
                              style={{
                                color: item.color,
                                opacity: currentOpacity,
                              }}
                            >
                              {item.value}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  );
                }}
              />
              
              {/* First dataset error */}
              <Line 
                type="monotone" 
                dataKey="firstError" 
                stroke="#3b82f6" 
                strokeWidth={2}
                strokeOpacity={lineOpacity.firstEstimate}
                dot={{ r: 2, fill: '#3b82f6', fillOpacity: lineOpacity.firstEstimate }}
                activeDot={{ r: 4, fillOpacity: lineOpacity.firstEstimate }}
                connectNulls={false}
                name={`${firstName} error`}
              />
              
              {/* Second dataset error */}
              <Line 
                type="monotone" 
                dataKey="secondError" 
                stroke="#7c3aed" 
                strokeWidth={2}
                strokeOpacity={lineOpacity.secondEstimate}
                dot={{ r: 2, fill: '#7c3aed', fillOpacity: lineOpacity.secondEstimate }}
                activeDot={{ r: 4, fillOpacity: lineOpacity.secondEstimate }}
                connectNulls={false}
                name={`${secondName} error`}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Bar Chart Comparison */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-800">Target vs. estimates comparison</h3>
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
          </div>
        </div>

        {/* Target Search and Pagination */}
        <div className="mb-4 p-4 bg-gray-50 rounded-lg">
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
              Found {filteredTargets.length} target{filteredTargets.length !== 1 ? 's' : ''} containing "{searchQuery}"
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

        {/* Bar Chart */}
        {barChartData.length > 0 && selectedEpoch !== null ? (
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart 
                data={barChartData} 
                margin={{ top: 10, right: 30, left: 20, bottom: 80 }}
                barGap={2}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="targetName" 
                  angle={-45}
                  textAnchor="end"
                  height={80}
                  tick={{ fontSize: 10 }}
                  interval={0}
                />
                <YAxis
                  tickFormatter={formatValueCompact}
                  tick={{ fontSize: 10 }}
                  label={{ value: 'Value', angle: -90, position: 'insideLeft', textAnchor: 'middle', fontSize: 12 }}
                />
                <Tooltip 
                  formatter={(value: any, name: string) => {
                    const labels = {
                      targetValue: 'Target value',
                      firstEstimate: `${firstName} Estimate`,
                      secondEstimate: `${secondName} Estimate`
                    };
                    return [formatValue(value), labels[name as keyof typeof labels] || name];
                  }}
                  labelFormatter={(label) => `Target: ${label}`}
                />
                <Legend 
                  verticalAlign="top"
                  height={0}
                  content={({ payload }: any) => {
                    const barItems = payload || [];
                    
                    return (
                      <div className="flex flex-col items-center space-y-2 pb-4">
                        {barItems.map((item: any, index: number) => (
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
                            <span className="text-sm font-medium" style={{ color: item.color }}>
                              {item.value}
                            </span>
                          </div>
                        ))}
                      </div>
                    );
                  }}
                />
                
                {/* Target Value Bars */}
                <Bar dataKey="targetValue" name="Target value" fill="#22c55e" />
                
                {/* First Dataset Estimates */}
                <Bar dataKey="firstEstimate" name={firstName} fill="#3b82f6" />
                
                {/* Second Dataset Estimates */}
                <Bar dataKey="secondEstimate" name={secondName} fill="#7c3aed" />
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
      </div>

    </div>
  );
}