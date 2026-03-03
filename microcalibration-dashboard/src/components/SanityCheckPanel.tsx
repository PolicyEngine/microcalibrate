'use client';

import { ValidationDataPoint } from '@/types/calibration';
import { AlertTriangle } from 'lucide-react';

interface SanityCheckPanelProps {
  data: ValidationDataPoint[];
}

export default function SanityCheckPanel({ data }: SanityCheckPanelProps) {
  const failures = data.filter(d => d.sanity_check === 'FAIL');

  if (failures.length === 0) {
    return (
      <div className="bg-green-50 border border-green-200 rounded-lg p-4">
        <h3 className="text-sm font-medium text-green-800">All sanity checks passed</h3>
        <p className="text-sm text-green-700 mt-1">
          {data.length} targets validated with no sanity failures.
        </p>
      </div>
    );
  }

  const byArea = new Map<string, ValidationDataPoint[]>();
  for (const f of failures) {
    const key = `${f.area_type}/${f.area_id}`;
    if (!byArea.has(key)) byArea.set(key, []);
    byArea.get(key)!.push(f);
  }

  return (
    <div className="bg-red-50 border border-red-200 rounded-lg p-6">
      <div className="flex items-center gap-2 mb-4">
        <AlertTriangle className="w-5 h-5 text-red-600" />
        <h3 className="text-lg font-semibold text-red-800">
          {failures.length} Sanity Check Failure{failures.length !== 1 ? 's' : ''}
        </h3>
      </div>

      <div className="space-y-4">
        {Array.from(byArea.entries()).map(([area, areaFailures]) => (
          <div key={area} className="bg-white/60 rounded-md p-3">
            <h4 className="text-sm font-semibold text-red-700 mb-2">{area}</h4>
            <div className="space-y-1">
              {areaFailures.map((f, i) => (
                <div key={i} className="text-sm text-red-600 flex justify-between">
                  <span className="font-mono">{f.variable}</span>
                  <span className="text-red-500 text-xs ml-4">{f.sanity_reason}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
