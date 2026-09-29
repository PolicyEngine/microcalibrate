import { describe, expect, it } from 'vitest';
import fc from 'fast-check';
import { CalibrationDataPoint } from '@/types/calibration';
import { getEpochsNewestFirst, resolveSelectedEpoch } from '@/utils/epochs';

const point = (epoch: number, target_name = 't'): CalibrationDataPoint => ({
  epoch,
  loss: 0,
  target_name,
  target: 1,
  estimate: 1,
  error: 0,
  abs_error: 0,
  rel_abs_error: 0,
});

const datasetArb = fc.array(
  fc.record({ epoch: fc.nat({ max: 50 }), target_name: fc.string({ maxLength: 3 }) }),
  { maxLength: 30 }
).map(rows => rows.map(r => point(r.epoch, r.target_name)));

const datasetsArb = fc.array(datasetArb, { maxLength: 3 });

// The effect this replaced: selected epoch starts null, and after each
// render an effect adopts the newest epoch if nothing is selected yet.
function effectModelEpoch(chosen: Array<number | null>, epochsNewestFirst: number[]): number | null {
  let selected: number | null = null;
  const settle = () => {
    if (epochsNewestFirst.length > 0 && selected === null) {
      selected = epochsNewestFirst[0];
    }
  };
  settle();
  chosen.forEach(pick => {
    if (pick !== null) selected = pick;
    settle();
  });
  return selected;
}

describe('getEpochsNewestFirst', () => {
  it('lists each epoch present in any dataset exactly once, newest first', () => {
    fc.assert(
      fc.property(datasetsArb, datasets => {
        const epochs = getEpochsNewestFirst(...datasets);
        for (let i = 1; i < epochs.length; i++) {
          expect(epochs[i - 1]).toBeGreaterThan(epochs[i]);
        }
        const present = new Set(datasets.flat().map(p => p.epoch));
        expect(new Set(epochs)).toEqual(present);
        expect(epochs.length).toBe(present.size);
      })
    );
  });

  it('ignores dataset order and row order', () => {
    fc.assert(
      fc.property(datasetsArb, datasets => {
        const reversed = [...datasets].reverse().map(d => [...d].reverse());
        expect(getEpochsNewestFirst(...reversed)).toEqual(getEpochsNewestFirst(...datasets));
      })
    );
  });

  it('is empty without data', () => {
    expect(getEpochsNewestFirst()).toEqual([]);
    expect(getEpochsNewestFirst([], [])).toEqual([]);
  });
});

describe('resolveSelectedEpoch', () => {
  it("returns the user's pick whenever there is one", () => {
    fc.assert(
      fc.property(fc.nat(), datasetsArb, (chosen, datasets) => {
        expect(resolveSelectedEpoch(chosen, getEpochsNewestFirst(...datasets))).toBe(chosen);
      })
    );
  });

  it('defaults to the newest epoch, or null when there are none', () => {
    fc.assert(
      fc.property(datasetsArb, datasets => {
        const all = datasets.flat().map(p => p.epoch);
        const expected = all.length > 0 ? Math.max(...all) : null;
        expect(resolveSelectedEpoch(null, getEpochsNewestFirst(...datasets))).toBe(expected);
      })
    );
  });

  it('shows epoch 0 when it is the only epoch', () => {
    expect(resolveSelectedEpoch(null, getEpochsNewestFirst([point(0)]))).toBe(0);
  });

  it('matches the effect-based default it replaced for any sequence of picks', () => {
    fc.assert(
      fc.property(
        datasetsArb,
        fc.array(fc.option(fc.nat({ max: 50 }), { nil: null }), { maxLength: 5 }),
        (datasets, picks) => {
          const epochs = getEpochsNewestFirst(...datasets);
          // The derived version stores only the latest real pick.
          const lastPick = picks.filter((p): p is number => p !== null).at(-1) ?? null;
          expect(resolveSelectedEpoch(lastPick, epochs)).toBe(effectModelEpoch(picks, epochs));
        }
      )
    );
  });
});
