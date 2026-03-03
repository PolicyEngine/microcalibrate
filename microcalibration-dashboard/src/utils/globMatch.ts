export function globMatch(pattern: string, text: string): boolean {
  if (!pattern.includes('*') && !pattern.includes('?')) {
    return text.toLowerCase().includes(pattern.toLowerCase());
  }

  const escaped = pattern
    .replace(/[.+^${}()|[\]\\]/g, '\\$&')
    .replace(/\*/g, '.*')
    .replace(/\?/g, '.');

  return new RegExp(`^${escaped}$`, 'i').test(text);
}

export function hasGlobChars(pattern: string): boolean {
  return pattern.includes('*') || pattern.includes('?');
}
