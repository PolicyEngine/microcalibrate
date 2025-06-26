/**
 * Utility functions for ordering target names alphanumerically
 * with hierarchical support for "/" separated parts
 */

/**
 * Natural alphanumeric comparison function that handles numbers correctly
 * e.g., "item2" comes before "item10"
 */
function naturalCompare(a: string, b: string): number {
  // Split strings into chunks of text and numbers
  const aChunks = a.match(/(\d+|\D+)/g) || [];
  const bChunks = b.match(/(\d+|\D+)/g) || [];
  
  const maxLength = Math.max(aChunks.length, bChunks.length);
  
  for (let i = 0; i < maxLength; i++) {
    const aChunk = aChunks[i] || '';
    const bChunk = bChunks[i] || '';
    
    // If both chunks are numbers, compare numerically
    if (/^\d+$/.test(aChunk) && /^\d+$/.test(bChunk)) {
      const aNum = parseInt(aChunk, 10);
      const bNum = parseInt(bChunk, 10);
      if (aNum !== bNum) {
        return aNum - bNum;
      }
    } else {
      // Otherwise, compare lexicographically (case-insensitive)
      const result = aChunk.toLowerCase().localeCompare(bChunk.toLowerCase());
      if (result !== 0) {
        return result;
      }
    }
  }
  
  return 0;
}

/**
 * Compare two target names hierarchically by "/" separated parts
 * Each part is compared alphanumerically
 */
export function compareTargetNames(a: string, b: string): number {
  const aParts = a.split('/');
  const bParts = b.split('/');
  
  const maxParts = Math.max(aParts.length, bParts.length);
  
  for (let i = 0; i < maxParts; i++) {
    const aPart = aParts[i] || '';
    const bPart = bParts[i] || '';
    
    const result = naturalCompare(aPart, bPart);
    if (result !== 0) {
      return result;
    }
  }
  
  return 0;
}

/**
 * Sort an array of target names using hierarchical alphanumeric ordering
 */
export function sortTargetNames(targetNames: string[]): string[] {
  return [...targetNames].sort(compareTargetNames);
}

/**
 * Get unique target names from calibration data and sort them
 */
export function getSortedUniqueTargets(data: Array<{ target_name: string }>): string[] {
  const uniqueTargets = Array.from(new Set(data.map(d => d.target_name)));
  return sortTargetNames(uniqueTargets);
}

/**
 * Sort targets with search relevance priority
 * Exact matches first, then starts with, then contains, all alphanumerically sorted within each group
 */
export function sortTargetsWithRelevance(targets: string[], searchQuery: string): string[] {
  if (!searchQuery.trim()) {
    return sortTargetNames(targets);
  }
  
  const queryLower = searchQuery.toLowerCase();
  
  const exactMatches: string[] = [];
  const startsWithMatches: string[] = [];
  const containsMatches: string[] = [];
  
  targets.forEach(target => {
    const targetLower = target.toLowerCase();
    if (targetLower === queryLower) {
      exactMatches.push(target);
    } else if (targetLower.startsWith(queryLower)) {
      startsWithMatches.push(target);
    } else if (targetLower.includes(queryLower)) {
      containsMatches.push(target);
    }
  });
  
  return [
    ...sortTargetNames(exactMatches),
    ...sortTargetNames(startsWithMatches),
    ...sortTargetNames(containsMatches)
  ];
}