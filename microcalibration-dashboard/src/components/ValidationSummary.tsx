'use client';

import { ValidationDataPoint } from '@/types/calibration';

interface ValidationSummaryProps {
  data: ValidationDataPoint[];
}

function summarize(points: ValidationDataPoint[]) {
  const n = points.length;
  if (n === 0) return { n: 0, within5: 0, within10: 0, within20: 0, sanityFails: 0 };
  const within5 = points.filter(d => d.rel_abs_error < 0.05).length;
  const within10 = points.filter(d => d.rel_abs_error < 0.10).length;
  const within20 = points.filter(d => d.rel_abs_error < 0.20).length;
  const sanityFails = points.filter(d => d.sanity_check === 'FAIL').length;
  return { n, within5, within10, within20, sanityFails };
}

function pct(num: number, den: number): string {
  if (den === 0) return '—';
  return `${((num / den) * 100).toFixed(1)}%`;
}

export default function ValidationSummary({ data }: ValidationSummaryProps) {
  const areas = new Set(data.map(d => `${d.area_type}/${d.area_id}`));
  const inTraining = data.filter(d => d.in_training);
  const outOfSample = data.filter(d => !d.in_training);

  const allStats = summarize(data);
  const trainStats = summarize(inTraining);
  const oosStats = summarize(outOfSample);

  const cards = [
    { label: 'Areas Validated', value: String(areas.size), color: 'blue' },
    { label: 'Total Targets', value: String(allStats.n), color: 'gray' },
    { label: '< 5% Error', value: pct(allStats.within5, allStats.n), color: 'green' },
    { label: '< 10% Error', value: pct(allStats.within10, allStats.n), color: 'emerald' },
    { label: '< 20% Error', value: pct(allStats.within20, allStats.n), color: 'yellow' },
    { label: 'Sanity Failures', value: String(allStats.sanityFails), color: allStats.sanityFails > 0 ? 'red' : 'green' },
  ];

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">Validation Summary</h2>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
        {cards.map(card => (
          <div key={card.label} className="bg-gray-50 rounded-lg p-3 text-center">
            <div className={`text-2xl font-bold text-${card.color}-600`}>{card.value}</div>
            <div className="text-xs text-gray-500 mt-1">{card.label}</div>
          </div>
        ))}
      </div>

      {inTraining.length > 0 && outOfSample.length > 0 && (() => {
        const trainGroups = new Set(inTraining.map(d => d.target_name.replace(/^state_\d+\//, ''))).size;
        const oosGroups = new Set(outOfSample.map(d => d.target_name.replace(/^state_\d+\//, ''))).size;
        return (
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-blue-50 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-blue-800 mb-2">
                In-Training ({trainGroups} target group{trainGroups !== 1 ? 's' : ''} &times; {areas.size} areas = {trainStats.n})
              </h3>
              <div className="text-sm text-blue-700 space-y-1">
                <div>&lt;5%: {pct(trainStats.within5, trainStats.n)}</div>
                <div>&lt;10%: {pct(trainStats.within10, trainStats.n)}</div>
                <div>&lt;20%: {pct(trainStats.within20, trainStats.n)}</div>
              </div>
            </div>
            <div className="bg-purple-50 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-purple-800 mb-2">
                Out-of-Sample ({oosGroups} target group{oosGroups !== 1 ? 's' : ''} &times; {areas.size} areas = {oosStats.n})
              </h3>
              <div className="text-sm text-purple-700 space-y-1">
                <div>&lt;5%: {pct(oosStats.within5, oosStats.n)}</div>
                <div>&lt;10%: {pct(oosStats.within10, oosStats.n)}</div>
                <div>&lt;20%: {pct(oosStats.within20, oosStats.n)}</div>
              </div>
            </div>
          </div>
        );
      })()}
    </div>
  );
}
