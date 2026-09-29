import { describe, expect, it, vi } from 'vitest';
import fc from 'fast-check';
import type { ReactElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { colors } from '@policyengine/design-system/tokens/colors';
import ToggleLegend, { LegendPayloadItem } from '@/components/ToggleLegend';

type EntryElement = ReactElement<{ className: string; onClick: () => void }>;

// Legend entries as rendered: one clickable element per payload item.
function renderEntries(props: Parameters<typeof ToggleLegend>[0]): EntryElement[] {
  const root = ToggleLegend(props) as ReactElement<{ children?: EntryElement[] }>;
  return root.props.children ?? [];
}

// Distinct series ids, each with a random visibility, and a suffix that the
// chart appends to every series id to form its dataKey.
const legendArb = fc.record({
  series: fc.uniqueArray(fc.constantFrom('first', 'second', 'totalLoss', 'avgRelAbsError'), {
    minLength: 1,
  }),
  visibility: fc.array(fc.boolean(), { minLength: 4, maxLength: 4 }),
  suffix: fc.constantFrom('', 'TotalLoss', 'AvgError'),
});

describe('ToggleLegend', () => {
  it('toggles the series named by each entry, with the suffix removed', () => {
    fc.assert(
      fc.property(legendArb, ({ series, visibility, suffix }) => {
        const onToggleSeries = vi.fn();
        const payload: LegendPayloadItem[] = series.map(s => ({
          dataKey: `${s}${suffix}`,
          color: '#000000',
          value: s,
        }));
        const visibleSeries = Object.fromEntries(series.map((s, i) => [s, visibility[i]]));
        const entries = renderEntries({ payload, visibleSeries, onToggleSeries, seriesKeySuffix: suffix });

        expect(entries).toHaveLength(series.length);
        entries.forEach((entry, i) => {
          expect(entry.props.className).toContain(visibility[i] ? 'opacity-100' : 'opacity-50');
          entry.props.onClick();
          expect(onToggleSeries).toHaveBeenLastCalledWith(series[i]);
        });
        expect(onToggleSeries).toHaveBeenCalledTimes(series.length);
      })
    );
  });

  it('draws visible series in their own color and hidden ones in gray', () => {
    const html = renderToStaticMarkup(
      <ToggleLegend
        payload={[
          { dataKey: 'firstTotalLoss', color: '#123456', value: 'Run A' },
          { dataKey: 'secondTotalLoss', color: '#654321', value: 'Run B' },
        ]}
        visibleSeries={{ first: true, second: false }}
        onToggleSeries={() => {}}
        seriesKeySuffix="TotalLoss"
      />
    );
    expect(html).toContain('background-color:#123456');
    expect(html).not.toContain('#654321');
    expect(html).toContain(`background-color:${colors.gray[300]}`);
    expect(html).toContain('Run A');
    expect(html).toContain('Run B');
  });

  it('renders nothing but the container before Recharts supplies a payload', () => {
    expect(renderEntries({ visibleSeries: {}, onToggleSeries: () => {} })).toEqual([]);
  });
});
