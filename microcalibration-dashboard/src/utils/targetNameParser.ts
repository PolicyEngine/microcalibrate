export const FIPS_TO_STATE: Record<number, string> = {
  1: 'AL', 2: 'AK', 4: 'AZ', 5: 'AR', 6: 'CA', 8: 'CO', 9: 'CT',
  10: 'DE', 11: 'DC', 12: 'FL', 13: 'GA', 15: 'HI', 16: 'ID', 17: 'IL',
  18: 'IN', 19: 'IA', 20: 'KS', 21: 'KY', 22: 'LA', 23: 'ME', 24: 'MD',
  25: 'MA', 26: 'MI', 27: 'MN', 28: 'MS', 29: 'MO', 30: 'MT', 31: 'NE',
  32: 'NV', 33: 'NH', 34: 'NJ', 35: 'NM', 36: 'NY', 37: 'NC', 38: 'ND',
  39: 'OH', 40: 'OK', 41: 'OR', 42: 'PA', 44: 'RI', 45: 'SC', 46: 'SD',
  47: 'TN', 48: 'TX', 49: 'UT', 50: 'VT', 51: 'VA', 53: 'WA', 54: 'WV',
  55: 'WI', 56: 'WY',
};

export interface ParsedTargetName {
  geoPrefix: string;
  geoLevel: 'district' | 'state' | 'national' | 'unknown';
  stateFips: number | null;
  variable: string;
  constraint: string;
}

export function parseTargetName(targetName: string): ParsedTargetName {
  const parts = targetName.split('/');

  if (parts.length < 2) {
    return {
      geoPrefix: '',
      geoLevel: 'national',
      stateFips: null,
      variable: targetName,
      constraint: '',
    };
  }

  const geoPrefix = parts[0];
  const variable = parts[1];
  const constraint = parts.slice(2).join('/');

  let geoLevel: ParsedTargetName['geoLevel'] = 'unknown';
  let stateFips: number | null = null;

  if (geoPrefix.startsWith('cd_')) {
    geoLevel = 'district';
    const digits = geoPrefix.replace('cd_', '');
    stateFips = digits.length >= 2 ? parseInt(digits.substring(0, 2), 10) : null;
  } else if (geoPrefix.startsWith('state_')) {
    geoLevel = 'state';
    const digits = geoPrefix.replace('state_', '');
    stateFips = digits ? parseInt(digits, 10) : null;
  } else if (geoPrefix === 'national' || geoPrefix === 'us') {
    geoLevel = 'national';
  }

  return { geoPrefix, geoLevel, stateFips, variable, constraint };
}

export function variableKey(parsed: ParsedTargetName): string {
  return parsed.constraint
    ? `${parsed.variable}/${parsed.constraint}`
    : parsed.variable;
}
