'use client';

import { CalibrationDataPoint } from '@/types/calibration';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

interface RegressionAnalysisProps {
  firstData: CalibrationDataPoint[];
  secondData: CalibrationDataPoint[];
  firstName: string;
  secondName: string;
}

interface ScatterDataPoint {
  target_name: string;
  firstError: number;
  secondError: number;
  x: number; // firstError in percentage
  y: number; // secondError in percentage
}

interface RegressionStats {
  correlation: number;
  rSquared: number;
  slope: number;
  intercept: number;
  meanFirstError: number;
  meanSecondError: number;
}

export default function RegressionAnalysis({ firstData, secondData }: RegressionAnalysisProps) {
  // Get final epoch data for both datasets
  const getMaxEpoch = (data: CalibrationDataPoint[]) => {
    let maxEpoch = -Infinity;
    for (const point of data) {
      if (point.epoch > maxEpoch) {
        maxEpoch = point.epoch;
      }
    }
    return maxEpoch;
  };
  const firstMaxEpoch = getMaxEpoch(firstData);
  const secondMaxEpoch = getMaxEpoch(secondData);
  
  const firstFinalData = firstData.filter(d => d.epoch === firstMaxEpoch);
  const secondFinalData = secondData.filter(d => d.epoch === secondMaxEpoch);

  // Create scatter plot data for overlapping targets
  const scatterData: ScatterDataPoint[] = [];
  
  firstFinalData.forEach(firstTarget => {
    const secondTarget = secondFinalData.find(s => s.target_name === firstTarget.target_name);
    
    if (secondTarget && 
        firstTarget.rel_abs_error !== undefined && 
        secondTarget.rel_abs_error !== undefined &&
        !isNaN(firstTarget.rel_abs_error) && 
        !isNaN(secondTarget.rel_abs_error)) {
      
      const firstErrorPct = firstTarget.rel_abs_error * 100;
      const secondErrorPct = secondTarget.rel_abs_error * 100;
      
      scatterData.push({
        target_name: firstTarget.target_name,
        firstError: firstTarget.rel_abs_error,
        secondError: secondTarget.rel_abs_error,
        x: firstErrorPct,
        y: secondErrorPct
      });
    }
  });

  if (scatterData.length === 0) {
    return (
      <div className="bg-white border border-gray-300 p-6 rounded-lg shadow-sm">
        <h2 className="text-xl font-bold text-gray-800 mb-4">📈 Performance Correlation Analysis</h2>
        <div className="bg-yellow-50 border border-yellow-200 rounded p-4">
          <p className="text-yellow-800">
            No overlapping targets found between datasets for comparison. 
            This could mean the targets are completely different or there are data quality issues.
          </p>
        </div>
      </div>
    );
  }

  // Calculate regression statistics
  const calculateRegression = (data: ScatterDataPoint[]): RegressionStats => {
    const n = data.length;
    const sumX = data.reduce((sum, d) => sum + d.x, 0);
    const sumY = data.reduce((sum, d) => sum + d.y, 0);
    const sumXY = data.reduce((sum, d) => sum + d.x * d.y, 0);
    const sumXX = data.reduce((sum, d) => sum + d.x * d.x, 0);
    const sumYY = data.reduce((sum, d) => sum + d.y * d.y, 0);
    
    const meanX = sumX / n;
    const meanY = sumY / n;
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = meanY - slope * meanX;
    
    const correlation = (n * sumXY - sumX * sumY) / 
      Math.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));
    
    const rSquared = correlation * correlation;
    
    return {
      correlation,
      rSquared,
      slope,
      intercept,
      meanFirstError: meanX,
      meanSecondError: meanY
    };
  };

  const stats = calculateRegression(scatterData);

  // Calculate categorical analysis for summary
  const betterCount = scatterData.filter(d => d.y < d.x).length;
  const worseCount = scatterData.filter(d => d.y > d.x).length;
  const sameCount = scatterData.filter(d => Math.abs(d.y - d.x) < 0.1).length;

  // Calculate domain for scatter plot
  const allErrors = scatterData.flatMap(d => [d.x, d.y]);
  const maxError = Math.max(...allErrors);
  const minError = Math.min(...allErrors);
  const domain = [Math.max(0, minError - 1), maxError + 1];


  const formatTooltip = (value: number, name: string) => {
    if (name === 'y') {
      return [`${value.toPrecision(3)}%`, 'Second dataset error'];
    }
    return [`${value.toPrecision(3)}%`, 'First dataset error'];
  };

  const formatTickLabel = (value: number) => {
    return value.toPrecision(3);
  };

  const formatLabel = (label: string, payload: unknown) => {
    if (Array.isArray(payload) && payload.length > 0 && payload[0] && typeof payload[0] === 'object' && 'payload' in payload[0]) {
      const item = payload[0] as {payload: ScatterDataPoint};
      return `Target: ${item.payload.target_name}`;
    }
    return label;
  };

  return (
    <div className="bg-white border border-gray-300 p-6 rounded-lg shadow-sm">
      <h2 className="text-xl font-bold text-gray-800 mb-4">📈 Performance correlation analysis</h2>
      <p className="text-gray-600 mb-6">
        Scatter plot analysis of final epoch error rates for {scatterData.length} overlapping targets
      </p>

      {/* Statistics Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white border-2 border-gray-300 rounded-lg p-3 text-center shadow-sm">
          <div className="text-lg font-bold text-gray-800">{stats.correlation.toFixed(3)}</div>
          <div className="text-xs text-gray-600">Correlation (r)</div>
        </div>
        <div className="bg-white border-2 border-gray-300 rounded-lg p-3 text-center shadow-sm">
          <div className="text-lg font-bold text-gray-800">{worseCount}</div>
          <div className="text-xs text-gray-600">Better in 1st</div>
        </div>
        <div className="bg-white border-2 border-gray-300 rounded-lg p-3 text-center shadow-sm">
          <div className="text-lg font-bold text-gray-800">{betterCount}</div>
          <div className="text-xs text-gray-600">Better in 2nd</div>
        </div>
        <div className="bg-white border-2 border-gray-300 rounded-lg p-3 text-center shadow-sm">
          <div className="text-lg font-bold text-gray-800">{stats.slope.toFixed(3)}</div>
          <div className="text-xs text-gray-600">Slope</div>
        </div>
      </div>

      {/* Scatter Plot */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-3">Error comparison scatter plot</h3>
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 30, bottom: 40, left: 40 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                type="number" 
                dataKey="x" 
                domain={domain}
                label={{ value: 'First dataset rel abs error (%)', position: 'insideBottom', offset: -10, fontSize: 15 }}
                tick={{ fontSize: 10 }}
                tickFormatter={formatTickLabel}
              />
              <YAxis 
                type="number" 
                dataKey="y" 
                domain={domain}
                label={{ value: 'Second dataset rel abs error (%)', angle: -90, position: 'insideLeft', textAnchor: 'middle', dy: 145, fontSize: 15}}
                tick={{ fontSize: 10 }}
                tickFormatter={formatTickLabel}
              />
              <Tooltip 
                formatter={formatTooltip}
                labelFormatter={formatLabel}
                cursor={{ strokeDasharray: '3 3' }}
              />
              
              {/* Equality line (y = x) */}
              <ReferenceLine 
                segment={[{ x: domain[0], y: domain[0] }, { x: domain[1], y: domain[1] }]}
                stroke="#94a3b8" 
                strokeDasharray="5 5"
                strokeWidth={2}
              />
              
              {/* Data points */}
              <Scatter 
                name="Targets" 
                data={scatterData} 
                fill="#3b82f6"
                fillOpacity={0.6}
                r={3}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-2 text-xs text-gray-600 flex items-center justify-center space-x-4">
          <div className="flex items-center">
            <div className="w-3 h-0.5 bg-gray-400 mr-1"></div>
            <span>Equality line (y = x)</span>
          </div>
        </div>
      </div>

      {/* Interpretation */}
      <div className="bg-gray-50 border border-gray-200 rounded p-4">
        <h4 className="font-semibold text-gray-800 mb-3">Statistical interpretation</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <p className="text-gray-700 mb-2">
              <strong>Correlation:</strong> {stats.correlation > 0.7 ? 'Strong positive' : stats.correlation > 0.3 ? 'Moderate positive' : stats.correlation > -0.3 ? 'Weak' : stats.correlation > -0.7 ? 'Moderate negative' : 'Strong negative'} similarity (r = {stats.correlation.toFixed(3)}). Higher correlation indicates the two datasets have similar error patterns across targets.
            </p>
            <p className="text-gray-700">
              <strong>Slope:</strong> {stats.slope > 1.1 ? 'Second dataset errors increase faster' : stats.slope < 0.9 ? 'Second dataset has lower error growth' : 'Similar error scaling'} (slope = {stats.slope.toFixed(3)})
            </p>
          </div>
          <div>
            <p className="text-gray-700 mb-2">
              <strong>Performance change:</strong> When moving from the first to the second dataset, {betterCount} targets improved, {worseCount} worsened, {sameCount} remained similar
            </p>
            <p className="text-gray-700">
              <strong>Average errors:</strong> {stats.meanFirstError.toFixed(2)}% → {stats.meanSecondError.toFixed(2)}%
            </p>
          </div>
        </div>
      </div>

      {/* Overall Trend - Prominent Box */}
      <div className="mt-6 bg-blue-50 border-2 border-blue-200 rounded-lg p-6">
        <div className="text-center">
          <h4 className="text-lg font-bold text-blue-900 mb-2">Overall performance trend</h4>
          <p className="text-blue-800 text-lg">
            {stats.meanSecondError < stats.meanFirstError ? '📈 Second dataset performs better on average' : stats.meanSecondError > stats.meanFirstError ? '📉 First dataset performs better on average' : '➖ Similar average performance'}
          </p>
          <p className="text-blue-700 text-sm mt-2">
            Average error change: {stats.meanFirstError.toFixed(2)}% → {stats.meanSecondError.toFixed(2)}%
          </p>
        </div>
      </div>
    </div>
  );
}