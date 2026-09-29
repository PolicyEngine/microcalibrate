import { describe, expect, it } from 'vitest';
import fc from 'fast-check';
import { renderToString } from 'react-dom/server';
import {
  decodeDeeplink,
  encodeDeeplink,
  GitHubArtifactInfo,
  useUrlDeeplinkParams,
} from '@/utils/deeplinks';

const nonEmpty = fc.string({ minLength: 1, maxLength: 20, unit: 'binary' });

const artifactArb: fc.Arbitrary<GitHubArtifactInfo> = fc.record({
  repo: nonEmpty,
  branch: nonEmpty,
  commit: nonEmpty,
  artifact: nonEmpty,
});

const decodeQuery = (query: string) => decodeDeeplink(new URLSearchParams(query));

describe('deeplink encoding', () => {
  it('round-trips a single-artifact deeplink', () => {
    fc.assert(
      fc.property(artifactArb, primary => {
        const params = { mode: 'single' as const, primary };
        expect(decodeQuery(encodeDeeplink(params))).toEqual(params);
      })
    );
  });

  it('round-trips a comparison deeplink', () => {
    fc.assert(
      fc.property(artifactArb, artifactArb, (primary, secondary) => {
        const params = { mode: 'comparison' as const, primary, secondary };
        expect(decodeQuery(encodeDeeplink(params))).toEqual(params);
      })
    );
  });

  it('rejects a deeplink missing any required field', () => {
    fc.assert(
      fc.property(
        artifactArb,
        artifactArb,
        fc.boolean(),
        fc.nat(),
        (primary, secondary, comparison, drop) => {
          const params = comparison
            ? { mode: 'comparison' as const, primary, secondary }
            : { mode: 'single' as const, primary };
          const query = new URLSearchParams(encodeDeeplink(params));
          const required = [...query.keys()].filter(key => key !== 'mode');
          query.delete(required[drop % required.length]);
          expect(decodeDeeplink(query)).toBeNull();
        }
      )
    );
  });
});

describe('useUrlDeeplinkParams', () => {
  it('reads no deeplink while prerendering, so static HTML never depends on the URL', () => {
    function Probe() {
      return <span>{JSON.stringify(useUrlDeeplinkParams())}</span>;
    }
    expect(renderToString(<Probe />)).toBe('<span>null</span>');
  });
});
