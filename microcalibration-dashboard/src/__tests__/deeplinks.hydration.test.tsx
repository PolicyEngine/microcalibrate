// @vitest-environment happy-dom
import { describe, expect, it } from 'vitest';
import fc from 'fast-check';
import { act } from 'react';
import { renderToString } from 'react-dom/server';
import { hydrateRoot } from 'react-dom/client';
import { DeeplinkParams, encodeDeeplink, useUrlDeeplinkParams } from '@/utils/deeplinks';

// Lets act() flush React updates in tests.
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function Probe() {
  return <span>{JSON.stringify(useUrlDeeplinkParams())}</span>;
}

// Prerender Probe as the static export does, then hydrate it at `search`.
async function hydrateAt(search: string) {
  window.history.replaceState(null, '', `/${search}`);
  const container = document.createElement('div');
  container.innerHTML = renderToString(<Probe />);
  const prerendered = container.innerHTML;
  const hydrationErrors: unknown[] = [];
  const root = await act(async () =>
    hydrateRoot(container, <Probe />, { onRecoverableError: error => hydrationErrors.push(error) })
  );
  const hydrated = JSON.parse(container.textContent ?? '');
  await act(async () => root.unmount());
  return { prerendered, hydrated, hydrationErrors };
}

const nonEmpty = fc.string({ minLength: 1, maxLength: 12, unit: 'binary' });
const artifactArb = fc.record({ repo: nonEmpty, branch: nonEmpty, commit: nonEmpty, artifact: nonEmpty });
const paramsArb: fc.Arbitrary<DeeplinkParams> = fc.oneof(
  artifactArb.map(primary => ({ mode: 'single' as const, primary })),
  fc.tuple(artifactArb, artifactArb).map(([primary, secondary]) => ({
    mode: 'comparison' as const,
    primary,
    secondary,
  }))
);

describe('useUrlDeeplinkParams after hydration', () => {
  it('reads the deeplink in the URL without a hydration mismatch', async () => {
    await fc.assert(
      fc.asyncProperty(paramsArb, async params => {
        const { prerendered, hydrated, hydrationErrors } = await hydrateAt(`?${encodeDeeplink(params)}`);
        expect(prerendered).toBe('<span>null</span>');
        expect(hydrated).toEqual(params);
        expect(hydrationErrors).toEqual([]);
      }),
      { numRuns: 25 }
    );
  });

  it('reads no deeplink from a URL without one', async () => {
    for (const search of ['', '?repo=only-a-repo', '?mode=comparison&repo1=a']) {
      const { hydrated, hydrationErrors } = await hydrateAt(search);
      expect(hydrated).toBeNull();
      expect(hydrationErrors).toEqual([]);
    }
  });
});
