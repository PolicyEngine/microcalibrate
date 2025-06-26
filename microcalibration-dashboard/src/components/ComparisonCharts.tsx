'use client';

import { useState } from 'react';
import { CalibrationDataPoint } from '@/types/calibration';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { TrendingUp } from 'lucide-react';

interface ComparisonChartsProps {
  firstData: CalibrationDataPoint[];
  secondData: CalibrationDataPoint[];
  firstName: string;
  secondName: string;
}

export default function ComparisonCharts({ firstData, secondData, firstName, secondName }: ComparisonChartsProps) {
  const [visibleLossLines, setVisibleLossLines] = useState({
    first: true,
    second: true
  });

  const [visibleErrorLines, setVisibleErrorLines] = useState({
    first: true,
    second: true
  });

  // Prepare chart data for both datasets
  const prepareChartData = (data: CalibrationDataPoint[]) => {
    return data.reduce((acc, point) => {
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
  };

  const firstChartData = prepareChartData(firstData);
  const secondChartData = prepareChartData(secondData);

  // Merge data for combined chart (union of all epochs)
  const allEpochs = Array.from(new Set([
    ...firstChartData.map(d => d.epoch),
    ...secondChartData.map(d => d.epoch)
  ])).sort((a, b) => a - b);

  const combinedLossData = allEpochs.map(epoch => {
    const firstPoint = firstChartData.find(d => d.epoch === epoch);
    const secondPoint = secondChartData.find(d => d.epoch === epoch);
    return {
      epoch,
      firstTotalLoss: firstPoint?.totalLoss || null,
      secondTotalLoss: secondPoint?.totalLoss || null
    };
  });

  const combinedErrorData = allEpochs.map(epoch => {
    const firstPoint = firstChartData.find(d => d.epoch === epoch);
    const secondPoint = secondChartData.find(d => d.epoch === epoch);
    return {
      epoch,
      firstAvgError: firstPoint?.avgRelAbsError || null,
      secondAvgError: secondPoint?.avgRelAbsError || null
    };
  });

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

  const handleLossLegendClick = (dataKey: string) => {
    setVisibleLossLines(prev => ({
      ...prev,
      [dataKey]: !prev[dataKey as keyof typeof prev]
    }));
  };

  const handleErrorLegendClick = (dataKey: string) => {
    setVisibleErrorLines(prev => ({
      ...prev,
      [dataKey]: !prev[dataKey as keyof typeof prev]
    }));
  };

  const LossCustomLegend = (props: { payload?: Array<{ dataKey: string; color: string; value: string }> }) => {
    const { payload } = props;
    return (
      <div className="flex justify-center items-center space-x-6 pt-4">
        {payload?.map((entry, index: number) => {
          const isVisible = visibleLossLines[entry.dataKey.replace('TotalLoss', '') as keyof typeof visibleLossLines];
          return (
            <div
              key={`loss-legend-${index}`}
              className={`flex items-center cursor-pointer ${
                isVisible ? 'opacity-100' : 'opacity-50'
              }`}
              onClick={() => handleLossLegendClick(entry.dataKey.replace('TotalLoss', ''))}
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

  const ErrorCustomLegend = (props: { payload?: Array<{ dataKey: string; color: string; value: string }> }) => {
    const { payload } = props;
    return (
      <div className="flex justify-center items-center space-x-6 pt-4">
        {payload?.map((entry, index: number) => {
          const isVisible = visibleErrorLines[entry.dataKey.replace('AvgError', '') as keyof typeof visibleErrorLines];
          return (
            <div
              key={`error-legend-${index}`}
              className={`flex items-center cursor-pointer ${
                isVisible ? 'opacity-100' : 'opacity-50'
              }`}
              onClick={() => handleErrorLegendClick(entry.dataKey.replace('AvgError', ''))}
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

  if (firstChartData.length === 0 && secondChartData.length === 0) {
    return null;
  }

  return (
    <div className="bg-white border border-gray-300 p-6 rounded-lg shadow-sm">
      <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
        <TrendingUp className="w-5 h-5 mr-2" />
        Loss and error progression comparison
      </h2>
      <p className="text-gray-600 mb-6">
        Comparison of total loss and average relative error over epochs between datasets
      </p>

      {/* Total Loss Chart */}
      {combinedLossData.length > 0 && (
        <div className="mb-8">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Total loss comparison</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={combinedLossData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="epoch" 
                  label={{ value: 'Epoch', position: 'insideBottom', offset: -10, fontSize: 12 }}
                  tick={{ fontSize: 10 }}
                />
                <YAxis
                  tickFormatter={formatLoss}
                  label={{ value: 'Total loss', angle: -90, position: 'insideLeft', textAnchor: 'middle', fontSize: 12, dy: 25 }}
                  tick={{ fontSize: 10 }}
                />
                <Tooltip 
                  formatter={(value: string | number | (string | number)[], name: string) => {
                    if (value === null || value === undefined) return ['N/A', name];
                    const numValue = Number(value);
                    if (name === 'firstTotalLoss') {
                      return [formatLoss(numValue), firstName];
                    } else if (name === 'secondTotalLoss') {
                      return [formatLoss(numValue), secondName];
                    }
                    return [formatLoss(numValue), name];
                  }}
                  labelFormatter={(label) => `Epoch: ${label}`}
                />
                <Legend content={<LossCustomLegend />} />
                <Line 
                  type="monotone" 
                  dataKey="firstTotalLoss" 
                  stroke={visibleLossLines.first ? "#2563eb" : "transparent"}
                  strokeWidth={2}
                  dot={visibleLossLines.first ? { r: 3 } : false}
                  activeDot={visibleLossLines.first ? { r: 5 } : false}
                  name={firstName}
                  hide={!visibleLossLines.first}
                  connectNulls={false}
                />
                <Line 
                  type="monotone" 
                  dataKey="secondTotalLoss" 
                  stroke={visibleLossLines.second ? "#7c3aed" : "transparent"}
                  strokeWidth={2}
                  dot={visibleLossLines.second ? { r: 3 } : false}
                  activeDot={visibleLossLines.second ? { r: 5 } : false}
                  name={secondName}
                  hide={!visibleLossLines.second}
                  connectNulls={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Average Relative Error Chart */}
      {combinedErrorData.length > 0 && (
        <div className="border-t pt-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Average relative error comparison</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={combinedErrorData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="epoch" 
                  label={{ value: 'Epoch', position: 'insideBottom', offset: -10, fontSize: 12 }}
                  tick={{ fontSize: 10 }}
                />
                <YAxis
                  tickFormatter={formatError}
                  label={{ value: 'Avg rel error (%)', angle: -90, position: 'insideLeft', textAnchor: 'middle', fontSize: 12, dy: 55 }}
                  tick={{ fontSize: 10 }}
                />
                <Tooltip 
                  formatter={(value: string | number | (string | number)[], name: string) => {
                    if (value === null || value === undefined) return ['N/A', name];
                    const numValue = Number(value);
                    if (name === 'firstAvgError') {
                      return [formatError(numValue), firstName];
                    } else if (name === 'secondAvgError') {
                      return [formatError(numValue), secondName];
                    }
                    return [formatError(numValue), name];
                  }}
                  labelFormatter={(label) => `Epoch: ${label}`}
                />
                <Legend content={<ErrorCustomLegend />} />
                <Line 
                  type="monotone" 
                  dataKey="firstAvgError" 
                  stroke={visibleErrorLines.first ? "#2563eb" : "transparent"}
                  strokeWidth={2}
                  dot={visibleErrorLines.first ? { r: 3 } : false}
                  activeDot={visibleErrorLines.first ? { r: 5 } : false}
                  name={firstName}
                  hide={!visibleErrorLines.first}
                  connectNulls={false}
                />
                <Line 
                  type="monotone" 
                  dataKey="secondAvgError" 
                  stroke={visibleErrorLines.second ? "#7c3aed" : "transparent"}
                  strokeWidth={2}
                  dot={visibleErrorLines.second ? { r: 3 } : false}
                  activeDot={visibleErrorLines.second ? { r: 5 } : false}
                  name={secondName}
                  hide={!visibleErrorLines.second}
                  connectNulls={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}