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

// Epoch 0 is drawn often: falsy epochs are where `||`-style defaults break.
const epochArb = fc.oneof(fc.constant(0), fc.nat({ max: 50 }));

const datasetArb = fc.array(
  fc.record({ epoch: epochArb, target_name: fc.string({ maxLength: 3 }) }),
  { maxLength: 30 }
).map(rows => rows.map(r => point(r.epoch, r.target_name)));

const datasetsArb = fc.array(datasetArb, { maxLength: 3 });

// The effect this replaced: selected epoch starts null, and after each
// render an effect adopts the newest epoch if nothing is selected yet. Models
// a chart whose data stays fixed while it is mounted.
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

  it('matches the effect-based default it replaced for any sequence of picks on fixed data', () => {
    fc.assert(
      fc.property(
        datasetsArb,
        fc.array(fc.option(epochArb, { nil: null }), { maxLength: 5 }),
        (datasets, picks) => {
          const epochs = getEpochsNewestFirst(...datasets);
          // The derived version stores only the latest real pick.
          const lastPick = picks.filter((p): p is number => p !== null).at(-1) ?? null;
          expect(resolveSelectedEpoch(lastPick, epochs)).toBe(effectModelEpoch(picks, epochs));
        }
      )
    );
  });

  // Intended change: the effect adopted the newest epoch once and kept it, so
  // if new data arrived while a chart stayed mounted (e.g. a load finishing
  // after "View dashboard"), it could point at an epoch the new data lacks.
  // The derived default follows the newest epoch of the current data.
  it('follows new data until the user picks an epoch', () => {
    const before = getEpochsNewestFirst([point(0), point(10)]);
    const after = getEpochsNewestFirst([point(0), point(5)]);
    expect(resolveSelectedEpoch(null, before)).toBe(10);
    expect(resolveSelectedEpoch(null, after)).toBe(5);
    expect(resolveSelectedEpoch(0, after)).toBe(0);
  });
});
