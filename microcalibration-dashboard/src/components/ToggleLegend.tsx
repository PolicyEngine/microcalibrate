'use client';

import { colors } from '@policyengine/design-system/tokens/colors';

export interface LegendPayloadItem {
  dataKey: string;
  color: string;
  value: string;
}

interface ToggleLegendProps {
  // Injected by Recharts when this element is a <Legend> `content`.
  payload?: LegendPayloadItem[];
  visibleSeries: Record<string, boolean>;
  onToggleSeries: (seriesKey: string) => void;
  // Removed from each entry's dataKey to get its key in `visibleSeries`,
  // e.g. 'TotalLoss' maps 'firstTotalLoss' to 'first'.
  seriesKeySuffix?: string;
}

// Legend whose entries show or hide their line when clicked. Declared at
// module scope so React keeps its identity across chart re-renders.
export default function ToggleLegend({
  payload,
  visibleSeries,
  onToggleSeries,
  seriesKeySuffix = '',
}: ToggleLegendProps) {
  return (
    <div className="flex justify-center items-center space-x-6 pt-4">
      {payload?.map((entry, index: number) => {
        const seriesKey = entry.dataKey.replace(seriesKeySuffix, '');
        const isVisible = visibleSeries[seriesKey];
        return (
          <div
            key={`legend-${index}`}
            className={`flex items-center cursor-pointer ${
              isVisible ? 'opacity-100' : 'opacity-50'
            }`}
            onClick={() => onToggleSeries(seriesKey)}
          >
            <div
              className="w-3 h-0.5 mr-2"
              style={{ backgroundColor: isVisible ? entry.color : colors.gray[300] }}
            />
            <span className="text-sm text-gray-700">{entry.value}</span>
          </div>
        );
      })}
    </div>
  );
}
