export interface GitHubArtifactInfo {
  repo: string;
  branch: string;
  commit: string;
  artifact: string;
}

export interface DeeplinkParams {
  mode?: 'single' | 'comparison';
  primary?: GitHubArtifactInfo | null;
  secondary?: GitHubArtifactInfo | null;
}

export function encodeDeeplink(params: DeeplinkParams): string {
  const urlParams = new URLSearchParams();
  
  if (params.mode === 'comparison' && params.primary && params.secondary) {
    urlParams.set('mode', 'comparison');
    
    urlParams.set('repo1', params.primary.repo);
    urlParams.set('branch1', params.primary.branch);
    urlParams.set('commit1', params.primary.commit);
    urlParams.set('artifact1', params.primary.artifact);
    
    urlParams.set('repo2', params.secondary.repo);
    urlParams.set('branch2', params.secondary.branch);
    urlParams.set('commit2', params.secondary.commit);
    urlParams.set('artifact2', params.secondary.artifact);
  } else if (params.primary) {
    urlParams.set('repo', params.primary.repo);
    urlParams.set('branch', params.primary.branch);
    urlParams.set('commit', params.primary.commit);
    urlParams.set('artifact', params.primary.artifact);
  }
  
  return urlParams.toString();
}

export function decodeDeeplink(searchParams: URLSearchParams): DeeplinkParams | null {
  const mode = searchParams.get('mode');
  
  if (mode === 'comparison') {
    const repo1 = searchParams.get('repo1');
    const branch1 = searchParams.get('branch1');
    const commit1 = searchParams.get('commit1');
    const artifact1 = searchParams.get('artifact1');
    
    const repo2 = searchParams.get('repo2');
    const branch2 = searchParams.get('branch2');
    const commit2 = searchParams.get('commit2');
    const artifact2 = searchParams.get('artifact2');
    
    if (repo1 && branch1 && commit1 && artifact1 && repo2 && branch2 && commit2 && artifact2) {
      return {
        mode: 'comparison',
        primary: { repo: repo1, branch: branch1, commit: commit1, artifact: artifact1 },
        secondary: { repo: repo2, branch: branch2, commit: commit2, artifact: artifact2 }
      };
    }
  } else {
    const repo = searchParams.get('repo');
    const branch = searchParams.get('branch');
    const commit = searchParams.get('commit');
    const artifact = searchParams.get('artifact');
    
    if (repo && branch && commit && artifact) {
      return {
        mode: 'single',
        primary: { repo, branch, commit, artifact }
      };
    }
  }
  
  return null;
}

export function generateShareableUrl(params: DeeplinkParams): string {
  const encoded = encodeDeeplink(params);
  const baseUrl = typeof window !== 'undefined' 
    ? `${window.location.protocol}//${window.location.host}${window.location.pathname}`
    : '';
  
  return encoded ? `${baseUrl}?${encoded}` : baseUrl;
}

export function getCurrentDeeplinkParams(): DeeplinkParams | null {
  if (typeof window === 'undefined') return null;
  
  const searchParams = new URLSearchParams(window.location.search);
  return decodeDeeplink(searchParams);
}