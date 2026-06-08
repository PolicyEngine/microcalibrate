'use client';

import { useMemo, useState } from 'react';
import { ValidationDataPoint } from '@/types/calibration';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts';

interface ValidationScatterPlotProps {
  data: ValidationDataPoint[];
}

function formatAxis(n: number): string {
  if (Math.abs(n) >= 1e12) return `${(n / 1e12).toFixed(1)}T`;
  if (Math.abs(n) >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (Math.abs(n) >= 1e3) return `${(n / 1e3).toFixed(0)}K`;
  return n.toFixed(0);
}

function logTick(lo: number, hi: number): number[] {
  const ticks: number[] = [];
  const minExp = Math.floor(Math.log10(Math.max(lo, 1)));
  const maxExp = Math.ceil(Math.log10(Math.max(hi, 10)));
  for (let e = minExp; e <= maxExp; e++) {
    ticks.push(Math.pow(10, e));
  }
  return ticks;
}

export default function ValidationScatterPlot({ data }: ValidationScatterPlotProps) {
  const [useLog, setUseLog] = useState(true);

  const points = useMemo(() => {
    return data
      .filter(d => isFinite(d.target_value) && isFinite(d.sim_value))
      .map(d => ({
        x: d.target_value,
        y: d.sim_value,
        variable: d.variable,
        area: d.area_id,
        inTraining: d.in_training,
        sanity: d.sanity_check,
      }));
  }, [data]);

  const logPoints = useMemo(() => {
    return points.filter(p => p.x > 0 && p.y > 0);
  }, [points]);

  if (points.length === 0) return null;

  const active = useLog ? logPoints : points;
  const allVals = active.flatMap(p => [p.x, p.y]);
  const minVal = Math.min(...allVals);
  const maxVal = Math.max(...allVals);

  let lo: number, hi: number, domain: [number, number];
  let ticks: number[] | undefined;

  if (useLog) {
    lo = Math.max(minVal * 0.5, 1);
    hi = maxVal * 2;
    domain = [lo, hi];
    ticks = logTick(lo, hi);
  } else {
    const pad = (maxVal - minVal) * 0.05;
    lo = minVal - pad;
    hi = maxVal + pad;
    domain = [lo, hi];
  }

  const inTraining = active.filter(p => p.inTraining);
  const outOfSample = active.filter(p => !p.inTraining);
  const skipped = useLog ? points.length - logPoints.length : 0;

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-gray-900">
          Sim value vs target value
        </h2>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setUseLog(!useLog)}
            className={`px-3 py-1 text-xs font-medium rounded ${
              useLog
                ? 'bg-blue-100 text-blue-700'
                : 'bg-gray-100 text-gray-600'
            }`}
          >
            {useLog ? 'Log scale' : 'Linear scale'}
          </button>
          {skipped > 0 && (
            <span className="text-xs text-gray-400">
              ({skipped} zero-value points hidden)
            </span>
          )}
        </div>
      </div>
      <ResponsiveContainer width="100%" height={500}>
        <ScatterChart margin={{ top: 10, right: 30, bottom: 30, left: 30 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            type="number" dataKey="x" name="Target Value"
            domain={domain}
            scale={useLog ? 'log' : 'auto'}
            ticks={ticks}
            tickFormatter={formatAxis}
            label={{ value: 'Target Value', position: 'bottom', offset: 10 }}
          />
          <YAxis
            type="number" dataKey="y" name="Sim Value"
            domain={domain}
            scale={useLog ? 'log' : 'auto'}
            ticks={ticks}
            tickFormatter={formatAxis}
            label={{ value: 'Sim Value', angle: -90, position: 'insideLeft', offset: -10 }}
          />
          <Tooltip
            formatter={(value: number, name: string) => [formatAxis(value), name]}
            content={({ active: isActive, payload }) => {
              if (!isActive || !payload?.length) return null;
              const p = payload[0].payload;
              return (
                <div className="bg-white border border-gray-200 rounded-md shadow-md p-2 text-xs">
                  <div className="font-semibold">{p.variable}</div>
                  <div>Area: {p.area}</div>
                  <div>Target: {formatAxis(p.x)}</div>
                  <div>Sim: {formatAxis(p.y)}</div>
                  <div>{p.inTraining ? 'In-training' : 'Out-of-sample'}</div>
                </div>
              );
            }}
          />
          <ReferenceLine
            segment={[{ x: domain[0], y: domain[0] }, { x: domain[1], y: domain[1] }]}
            stroke="#9ca3af" strokeDasharray="5 5" strokeWidth={1.5}
          />
          {inTraining.length > 0 && (
            <Scatter name="In-training" data={inTraining} fill="#3b82f6" opacity={0.6} />
          )}
          {outOfSample.length > 0 && (
            <Scatter name="Out-of-sample" data={outOfSample} fill="#a855f7" opacity={0.6} />
          )}
        </ScatterChart>
      </ResponsiveContainer>
      <div className="flex justify-center gap-6 mt-2 text-xs text-gray-500">
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded-full bg-blue-500" /> In-training
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded-full bg-purple-500" /> Out-of-sample
        </span>
        <span>Dashed line = perfect agreement</span>
      </div>
    </div>
  );
}
