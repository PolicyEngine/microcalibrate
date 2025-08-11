'use client';

import { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { CalibrationDataPoint } from '@/types/calibration';
import { getSortedUniqueTargets, sortTargetsWithRelevance } from '@/utils/targetOrdering';

interface LossChartProps {
  data: CalibrationDataPoint[];
}

export default function LossChart({ data }: LossChartProps) {
  const targetNames = getSortedUniqueTargets(data);
  const [selectedTarget, setSelectedTarget] = useState<string>(targetNames[0] || '');
  const [targetSearchQuery, setTargetSearchQuery] = useState<string>('');
  const [showTargetDropdown, setShowTargetDropdown] = useState<boolean>(false);

  // Filter targets based on search query
  const searchFilteredTargets = sortTargetsWithRelevance(
    targetNames.filter(target =>
      target.toLowerCase().includes(targetSearchQuery.toLowerCase())
    ),
    targetSearchQuery
  );

  if (targetNames.length === 0) {
    return (
      <div className="bg-white p-6 rounded-lg shadow border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Target Convergence</h3>
        <p className="text-gray-600">No data available</p>
      </div>
    );
  }

  // Filter data for selected target and prepare chart data
  const targetData = data
    .filter(d => d.target_name === selectedTarget)
    .sort((a, b) => a.epoch - b.epoch)
    .map(d => ({
      epoch: d.epoch,
      target: d.target,
      estimate: d.estimate,
      error: Math.abs(d.error),
      rel_error: d.rel_abs_error
    }));

  const yErrorTicks = (() => {
    const rels = targetData.map(d => d.rel_error).filter(Boolean);
    if (rels.length === 0) return [];
    const minExp = Math.floor(Math.log10(Math.min(...rels)));
    const maxExp = Math.ceil(Math.log10(Math.max(...rels)));
    return Array.from({ length: maxExp - minExp + 1 },
                      (_, i) => 10 ** (minExp + i));
  })();

  const formatValue = (value: number) => {
    if (Math.abs(value) >= 1000000) {
      return (value / 1000000).toFixed(1) + 'M';
    } else if (Math.abs(value) >= 1000) {
      return (value / 1000).toFixed(1) + 'K';
    }
    return value.toFixed(0);
  };

  const formatError = (value: number) => {
    if (value < 0.001) return value.toExponential(2);
    return value.toFixed(3);
  };

  // Get target value (should be constant)
  const targetValue = targetData[0]?.target || 0;
  const finalEstimate = targetData[targetData.length - 1]?.estimate || 0;
  const finalError = targetData[targetData.length - 1]?.rel_error || 0;

  // Calculate Y-axis domain to avoid empty space
  const allValues = targetData.flatMap(d => [d.target, d.estimate]);
  const minValue = Math.min(...allValues);
  const maxValue = Math.max(...allValues);
  const range = maxValue - minValue;
  const padding = range * 0.1; // 10% padding
  const yAxisDomain = [
    Math.max(0, minValue - padding),
    maxValue + padding
  ];

  return (
    <div className="bg-white p-6 rounded-lg shadow border">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Target convergence analysis</h3>
        <div className="flex items-center space-x-3">
          <label className="text-sm font-medium text-gray-700">
            Select target:
          </label>
          <div className="relative min-w-0" style={{ minWidth: '200px' }}>
            <input
              type="text"
              placeholder="Search targets..."
              value={targetSearchQuery}
              onChange={(e) => {
                setTargetSearchQuery(e.target.value);
                setShowTargetDropdown(true);
              }}
              onFocus={() => setShowTargetDropdown(true)}
              onBlur={() => setTimeout(() => setShowTargetDropdown(false), 150)}
              className="w-full border border-gray-300 rounded-md px-3 py-1 text-sm text-gray-900 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
            {selectedTarget && (
              <div className="mt-1 text-xs text-gray-600 truncate" title={selectedTarget}>
                Selected: {selectedTarget}
              </div>
            )}
            
            {/* Dropdown */}
            {showTargetDropdown && searchFilteredTargets.length > 0 && (
              <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg max-h-60 overflow-y-auto">
                {searchFilteredTargets.slice(0, 10).map(target => (
                  <div
                    key={target}
                    onClick={() => {
                      setSelectedTarget(target);
                      setTargetSearchQuery('');
                      setShowTargetDropdown(false);
                    }}
                    className={`px-3 py-2 text-sm cursor-pointer hover:bg-blue-50 ${
                      target === selectedTarget ? 'bg-blue-100 text-blue-800' : 'text-gray-900'
                    }`}
                    title={target}
                  >
                    <div className="truncate">{target}</div>
                  </div>
                ))}
                {searchFilteredTargets.length > 10 && (
                  <div className="px-3 py-2 text-xs text-gray-500 border-t">
                    And {searchFilteredTargets.length - 10} more... (refine search)
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Summary statistics */}
      <div className="grid grid-cols-3 gap-4 mb-6 p-3 bg-gray-50 rounded-lg">
        <div className="text-center">
          <div className="text-lg font-bold text-gray-900">{formatValue(targetValue)}</div>
          <div className="text-sm text-gray-600">Target value</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-gray-900">{formatValue(finalEstimate)}</div>
          <div className="text-sm text-gray-600">Final estimate</div>
        </div>
        <div className="text-center">
          <div className={`text-lg font-bold ${finalError < 0.05 ? 'text-green-600' : finalError < 0.20 ? 'text-yellow-600' : 'text-red-600'}`}>
            {(finalError * 100).toFixed(2)}%
          </div>
          <div className="text-sm text-gray-600">Final error</div>
        </div>
      </div>

      {/* Main chart */}
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={targetData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="epoch" 
              label={{ value: 'Epoch', position: 'insideBottom', offset: -10, fontSize: 12 }}
              tick={{ fontSize: 10 }}
            />
            <YAxis
            domain={yAxisDomain}
            tickFormatter={formatValue}
            tickCount={6}
            tickMargin={14}
            label={{ value: 'Value', angle: -90, position: 'insideLeft', textAnchor: 'middle', dx: -19, fontSize: 12 }}
            tick={{ fontSize: 10 }}
            />
            <Tooltip 
              formatter={(value: number, name: string) => [
                formatValue(value), 
                name === 'Target' ? 'Target' : 'Estimate'
              ]}
              labelFormatter={(label) => `Epoch: ${label}`}
            />
            <Legend wrapperStyle={{ paddingTop: '20px' }} />
            <Line 
              type="monotone" 
              dataKey="target" 
              stroke="#dc2626" 
              strokeWidth={3}
              strokeDasharray="5 5"
              dot={false}
              name="Target"
            />
            <Line 
              type="monotone" 
              dataKey="estimate" 
              stroke="#2563eb" 
              strokeWidth={2}
              dot={{ r: 2 }}
              activeDot={{ r: 4 }}
              name="Estimate"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Error chart */}
      <div className="mt-6">
        <h4 className="text-md font-semibold text-gray-800 mb-3">Relative error over time</h4>
        <div className="h-60">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={targetData} margin={{ top: 20, right: 30, left: 20, bottom: 50 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="epoch" 
                label={{ value: 'Epoch', position: 'insideBottom', offset: -10, fontSize: 12 }}
                tick={{ fontSize: 10 }}
              />
              <YAxis
              scale="log"
              domain={[yErrorTicks[0], yErrorTicks[yErrorTicks.length - 1]]}
              ticks={yErrorTicks}
              tickFormatter={formatError}
              tickMargin={14}
              label={{ value: 'Rel error (log)', angle: -90, position: 'insideLeft', textAnchor: 'start', dx: -19, dy: 60, fontSize: 12 }}
              tick={{ fontSize: 10 }}
              />
              <Tooltip 
                formatter={(value: number) => [formatError(value), 'Relative error']}
                labelFormatter={(label) => `Epoch: ${label}`}
              />
              <Line 
                type="monotone" 
                dataKey="rel_error" 
                stroke="#7c3aed" 
                strokeWidth={2}
                dot={{ r: 2 }}
                activeDot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}