'use client';

import { useMemo } from 'react';
import { ValidationDataPoint } from '@/types/calibration';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, Legend,
} from 'recharts';

interface ValidationByAreaChartProps {
  data: ValidationDataPoint[];
}

function getBarColor(value: number): string {
  if (value < 0.05) return '#16a34a';
  if (value < 0.10) return '#65a30d';
  if (value < 0.20) return '#ca8a04';
  if (value < 0.50) return '#ea580c';
  return '#dc2626';
}

export default function ValidationByAreaChart({ data }: ValidationByAreaChartProps) {
  const chartData = useMemo(() => {
    const byArea = new Map<string, number[]>();
    for (const d of data) {
      const key = d.area_id;
      if (!byArea.has(key)) byArea.set(key, []);
      if (isFinite(d.rel_abs_error)) {
        byArea.get(key)!.push(d.rel_abs_error);
      }
    }

    return Array.from(byArea.entries())
      .map(([area, errors]) => ({
        area,
        meanError: errors.reduce((a, b) => a + b, 0) / errors.length,
        medianError: [...errors].sort((a, b) => a - b)[Math.floor(errors.length / 2)] ?? 0,
        count: errors.length,
      }))
      .sort((a, b) => b.meanError - a.meanError);
  }, [data]);

  if (chartData.length === 0) return null;

  // For single area, show a variable-level breakdown instead
  if (chartData.length === 1) {
    const areaId = chartData[0].area;
    const varData = data
      .filter(d => d.area_id === areaId && isFinite(d.rel_abs_error))
      .map(d => ({
        variable: d.variable + (d.target_name.includes('[') ? d.target_name.slice(d.target_name.indexOf('[')) : ''),
        relAbsError: Math.min(d.rel_abs_error, 5),
        rawError: d.rel_abs_error,
      }))
      .sort((a, b) => b.relAbsError - a.relAbsError)
      .slice(0, 30);

    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-2">Relative error by target ({areaId})</h2>
        <p className="text-xs text-gray-500 mb-4">Top 30 targets by error. Capped at 500% for display.</p>
        <ResponsiveContainer width="100%" height={Math.max(400, varData.length * 24)}>
          <BarChart data={varData} layout="vertical" margin={{ left: 220, right: 30, top: 5, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              type="number"
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            />
            <YAxis type="category" dataKey="variable" width={210} tick={{ fontSize: 10 }} />
            <Tooltip
              formatter={(_value: number) => {
                return [`${(_value * 100).toFixed(1)}%`, 'Rel Abs Error'];
              }}
            />
            <Bar dataKey="relAbsError" name="Rel Abs Error">
              {varData.map((entry, idx) => (
                <Cell key={idx} fill={getBarColor(entry.rawError)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">Mean relative error by area</h2>
      <ResponsiveContainer width="100%" height={Math.max(300, chartData.length * 28)}>
        <BarChart data={chartData} layout="vertical" margin={{ left: 60, right: 20, top: 5, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            type="number"
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
          />
          <YAxis type="category" dataKey="area" width={50} tick={{ fontSize: 11 }} />
          <Tooltip
            formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, 'Mean Rel Abs Error']}
            labelFormatter={(label: string) => `Area: ${label}`}
          />
          <Legend />
          <Bar dataKey="meanError" name="Mean Rel Abs Error">
            {chartData.map((entry, idx) => (
              <Cell key={idx} fill={getBarColor(entry.meanError)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
