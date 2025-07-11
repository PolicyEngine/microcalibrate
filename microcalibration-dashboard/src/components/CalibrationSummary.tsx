'use client';

import { useState } from 'react';
import { CalibrationDataPoint } from '@/types/calibration';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

interface CalibrationSummaryProps {
  data: CalibrationDataPoint[];
}

interface TargetSummary {
  target_name: string;
  initialError: number;
  finalError: number;
  improvement: number;
  category: 'improved_significantly' | 'worsened' | 'minimal_change';
}

export default function CalibrationSummary({ data }: CalibrationSummaryProps) {
  const [visibleLines, setVisibleLines] = useState({
    totalLoss: true,
    avgRelAbsError: true
  });

  if (data.length === 0) {
    return null;
  }

  // Group data by target and calculate improvement metrics
  const targetSummaries: TargetSummary[] = [];
  const targetGroups = new Map<string, CalibrationDataPoint[]>();
  
  // Group by target name
  data.forEach(point => {
    if (!targetGroups.has(point.target_name)) {
      targetGroups.set(point.target_name, []);
    }
    targetGroups.get(point.target_name)!.push(point);
  });

  // Identify excluded targets (those with constant estimates across epochs)
  const excludedTargets: string[] = [];
  targetGroups.forEach((points, targetName) => {
    if (points.length > 1) {
      const estimates = points.map(p => p.estimate);
      const isConstant = estimates.every(est => Math.abs(est - estimates[0]) < 1e-6);
      if (isConstant) {
        excludedTargets.push(targetName);
      }
    }
  });

  // Calculate metrics for each target
  targetGroups.forEach((points, targetName) => {
    // Sort by epoch to get initial and final states
    const sortedPoints = points.sort((a, b) => a.epoch - b.epoch);
    const initialPoint = sortedPoints[0];
    const finalPoint = sortedPoints[sortedPoints.length - 1];
    
    if (initialPoint && finalPoint && 
        initialPoint.rel_abs_error !== undefined && finalPoint.rel_abs_error !== undefined &&
        !isNaN(initialPoint.rel_abs_error) && !isNaN(finalPoint.rel_abs_error) &&
        !excludedTargets.includes(targetName)) { // Exclude excluded targets from improvement calculations
      
      const initialError = initialPoint.rel_abs_error;
      const finalError = finalPoint.rel_abs_error;
      const improvement = initialError - finalError; // Positive = improvement, negative = worsened
      const relativeImprovement = initialError > 0 ? improvement / initialError : 0;
      
      let category: 'improved_significantly' | 'worsened' | 'minimal_change';
      
      if (relativeImprovement > 0.2) { // Improved by more than 20%
        category = 'improved_significantly';
      } else if (relativeImprovement < -0.1) { // Worsened by more than 10%
        category = 'worsened';
      } else {
        category = 'minimal_change';
      }
      
      targetSummaries.push({
        target_name: targetName,
        initialError,
        finalError,
        improvement,
        category
      });
    }
  });

  // Calculate statistics
  const totalTargets = targetSummaries.length;
  const improvedSignificantly = targetSummaries.filter(t => t.category === 'improved_significantly').length;
  const worsened = targetSummaries.filter(t => t.category === 'worsened').length;
  const minimalChange = targetSummaries.filter(t => t.category === 'minimal_change').length;

  const improvedPercentage = totalTargets > 0 ? (improvedSignificantly / totalTargets * 100) : 0;
  const worsenedPercentage = totalTargets > 0 ? (worsened / totalTargets * 100) : 0;
  const minimalPercentage = totalTargets > 0 ? (minimalChange / totalTargets * 100) : 0;

  // Get examples for each category
  const improvedExamples = targetSummaries
    .filter(t => t.category === 'improved_significantly')
    .sort((a, b) => b.improvement - a.improvement)
    .slice(0, 3);
    
  const worsenedExamples = targetSummaries
    .filter(t => t.category === 'worsened')
    .sort((a, b) => a.improvement - b.improvement)
    .slice(0, 3);
    
  const minimalExamples = targetSummaries
    .filter(t => t.category === 'minimal_change')
    .slice(0, 3);

  // Prepare chart data: aggregate total loss and total relative absolute error by epoch
  const chartData = data.reduce((acc, point) => {
    const existingEpoch = acc.find(item => item.epoch === point.epoch);
    if (existingEpoch) {
      existingEpoch.totalLoss += point.loss;
      if (point.rel_abs_error !== undefined && !isNaN(point.rel_abs_error)) {
        existingEpoch.totalRelAbsError += point.rel_abs_error;
        existingEpoch.validErrorCount += 1;
      }
    } else {
      acc.push({
        epoch: point.epoch,
        totalLoss: point.loss,
        totalRelAbsError: point.rel_abs_error !== undefined && !isNaN(point.rel_abs_error) ? point.rel_abs_error : 0,
        validErrorCount: point.rel_abs_error !== undefined && !isNaN(point.rel_abs_error) ? 1 : 0
      });
    }
    return acc;
  }, [] as Array<{ epoch: number; totalLoss: number; totalRelAbsError: number; validErrorCount: number }>)
  .map(item => ({
    epoch: item.epoch,
    totalLoss: item.totalLoss,
    avgRelAbsError: item.validErrorCount > 0 ? item.totalRelAbsError / item.validErrorCount : 0
  }))
  .sort((a, b) => a.epoch - b.epoch);

  const formatLoss = (value: number) => {
    if (value >= 1000000) {
      return (value / 1000000).toFixed(2) + 'M';
    } else if (value >= 1000) {
      return (value / 1000).toFixed(2) + 'K';
    }
    return value.toFixed(2);
  };

  const formatError = (value: number) => {
    return (value * 100).toFixed(2) + '%';
  };

  const handleLegendClick = (dataKey: string) => {
    setVisibleLines(prev => ({
      ...prev,
      [dataKey]: !prev[dataKey as keyof typeof prev]
    }));
  };

  const CustomLegend = (props: { payload?: Array<{ dataKey: string; color: string; value: string }> }) => {
    const { payload } = props;
    return (
      <div className="flex justify-center items-center space-x-6 pt-4">
        {payload?.map((entry, index: number) => {
          const isVisible = visibleLines[entry.dataKey as keyof typeof visibleLines];
          return (
            <div
              key={`legend-${index}`}
              className={`flex items-center cursor-pointer ${
                isVisible ? 'opacity-100' : 'opacity-50'
              }`}
              onClick={() => handleLegendClick(entry.dataKey)}
            >
              <div
                className="w-3 h-0.5 mr-2"
                style={{ backgroundColor: isVisible ? entry.color : '#ccc' }}
              />
              <span className="text-sm text-gray-700">{entry.value}</span>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="bg-white border border-gray-300 p-6 rounded-lg shadow-sm">
      <h2 className="text-xl font-bold text-gray-800 mb-4">Calibration progress summary</h2>
      <p className="text-gray-600 mb-4">
        Analysis of how calibration affected each target&apos;s accuracy from initial to final epoch
      </p>

      {/* Excluded Targets Info */}
      <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-blue-800">Excluded targets</h3>
            <p className="text-xs text-blue-600 mt-1">
              Targets excluded from calibration
            </p>
          </div>
          <div className="text-right">
            {excludedTargets.length > 0 ? (
              <>
                <div className="text-lg font-bold text-blue-700">{excludedTargets.length}</div>
                <div className="text-xs text-blue-600">
                  {excludedTargets.length <= 3 
                    ? excludedTargets.join(', ')
                    : `${excludedTargets.slice(0, 3).join(', ')}, +${excludedTargets.length - 3} more`}
                </div>
              </>
            ) : (
              <div className="text-sm text-blue-600">None</div>
            )}
          </div>
        </div>
      </div>

      {/* Summary Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        {/* Improved Significantly */}
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center mb-2">
            <TrendingUp className="w-5 h-5 text-green-600 mr-2" />
            <h3 className="text-lg font-semibold text-green-800">Significantly improved</h3>
          </div>
          <div className="text-2xl font-bold text-green-700">{improvedSignificantly}</div>
          <div className="text-sm text-green-600">
            {improvedPercentage.toFixed(1)}% of targets ({improvedSignificantly}/{totalTargets})
          </div>
          <div className="text-xs text-green-500 mt-1">
            Reduced error by &gt;20%
          </div>
        </div>

        {/* Worsened */}
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center mb-2">
            <TrendingDown className="w-5 h-5 text-red-600 mr-2" />
            <h3 className="text-lg font-semibold text-red-800">Worsened</h3>
          </div>
          <div className="text-2xl font-bold text-red-700">{worsened}</div>
          <div className="text-sm text-red-600">
            {worsenedPercentage.toFixed(1)}% of targets ({worsened}/{totalTargets})
          </div>
          <div className="text-xs text-red-500 mt-1">
            Increased error by &gt;10%
          </div>
        </div>

        {/* Minimal Change */}
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
          <div className="flex items-center mb-2">
            <Minus className="w-5 h-5 text-gray-600 mr-2" />
            <h3 className="text-lg font-semibold text-gray-800">Minimal change</h3>
          </div>
          <div className="text-2xl font-bold text-gray-700">{minimalChange}</div>
          <div className="text-sm text-gray-600">
            {minimalPercentage.toFixed(1)}% of targets ({minimalChange}/{totalTargets})
          </div>
          <div className="text-xs text-gray-500 mt-1">
            Error change within ±20%/±10%
          </div>
        </div>
      </div>

      {/* Detailed Examples */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Top improvements */}
        {improvedExamples.length > 0 && (
          <div>
            <h4 className="text-md font-semibold text-green-800 mb-3 flex items-center">
              <TrendingUp className="w-4 h-4 mr-1" />
              Top improvements
            </h4>
            <div className="space-y-2">
              {improvedExamples.map((target, i) => (
                <div key={i} className="bg-green-50 p-3 rounded border">
                  <div className="text-sm font-medium text-gray-800 truncate" title={target.target_name}>
                    {target.target_name}
                  </div>
                  <div className="text-xs text-green-700">
                    {(target.initialError * 100).toFixed(1)}% → {(target.finalError * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-green-600">
                    -{(target.improvement * 100).toFixed(1)}pp improvement
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Worsened cases */}
        {worsenedExamples.length > 0 && (
          <div>
            <h4 className="text-md font-semibold text-red-800 mb-3 flex items-center">
              <TrendingDown className="w-4 h-4 mr-1" />
              Worsened cases
            </h4>
            <div className="space-y-2">
              {worsenedExamples.map((target, i) => (
                <div key={i} className="bg-red-50 p-3 rounded border">
                  <div className="text-sm font-medium text-gray-800 truncate" title={target.target_name}>
                    {target.target_name}
                  </div>
                  <div className="text-xs text-red-700">
                    {(target.initialError * 100).toFixed(1)}% → {(target.finalError * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-red-600">
                    +{Math.abs(target.improvement * 100).toFixed(1)}pp worsened
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Stable cases */}
        {minimalExamples.length > 0 && (
          <div>
            <h4 className="text-md font-semibold text-gray-800 mb-3 flex items-center">
              <Minus className="w-4 h-4 mr-1" />
              Stable cases
            </h4>
            <div className="space-y-2">
              {minimalExamples.map((target, i) => (
                <div key={i} className="bg-gray-50 p-3 rounded border">
                  <div className="text-sm font-medium text-gray-800 truncate" title={target.target_name}>
                    {target.target_name}
                  </div>
                  <div className="text-xs text-gray-700">
                    {(target.initialError * 100).toFixed(1)}% → {(target.finalError * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-600">
                    {target.improvement >= 0 ? '-' : '+'}{Math.abs(target.improvement * 100).toFixed(1)}pp change
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Total Loss and Error Chart */}
      {chartData.length > 0 && (
        <div className="mt-8 border-t pt-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Total loss and average relative error over epochs</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="epoch" 
                  label={{ value: 'Epoch', position: 'insideBottom', offset: -10, fontSize: 12 }}
                  tick={{ fontSize: 10 }}
                />
                <YAxis
                  yAxisId="loss"
                  orientation="left"
                  tickFormatter={formatLoss}
                  label={{ value: 'Total loss', angle: -90, position: 'insideLeft', textAnchor: 'middle', fontSize: 12, dy: 25 }}
                  tick={{ fontSize: 10 }}
                />
                <YAxis
                  yAxisId="error"
                  orientation="right"
                  tickFormatter={formatError}
                  label={{ value: 'Avg rel error (%)', angle: 90, position: 'insideRight', textAnchor: 'middle', fontSize: 12, dy: 75 }}
                  tick={{ fontSize: 10 }}
                />
                <Tooltip 
                  formatter={(value: number, name: string) => {
                    if (name === 'totalLoss') {
                      return [formatLoss(value), 'Total loss'];
                    } else if (name === 'avgRelAbsError') {
                      return [formatError(value), 'Avg rel error'];
                    }
                    return [value, name];
                  }}
                  labelFormatter={(label) => `Epoch: ${label}`}
                />
                <Legend content={<CustomLegend />} />
                <Line 
                  yAxisId="loss"
                  type="monotone" 
                  dataKey="totalLoss" 
                  stroke={visibleLines.totalLoss ? "#dc2626" : "transparent"}
                  strokeWidth={2}
                  dot={visibleLines.totalLoss ? { r: 3 } : false}
                  activeDot={visibleLines.totalLoss ? { r: 5 } : false}
                  name="Total loss"
                  hide={!visibleLines.totalLoss}
                />
                <Line 
                  yAxisId="error"
                  type="monotone" 
                  dataKey="avgRelAbsError" 
                  stroke={visibleLines.avgRelAbsError ? "#7c3aed" : "transparent"}
                  strokeWidth={2}
                  dot={visibleLines.avgRelAbsError ? { r: 3 } : false}
                  activeDot={visibleLines.avgRelAbsError ? { r: 5 } : false}
                  name="Avg rel error"
                  hide={!visibleLines.avgRelAbsError}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}