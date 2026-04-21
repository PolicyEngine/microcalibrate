import { NextRequest, NextResponse } from 'next/server';

const GITHUB_API_BASE = 'https://api.github.com';
const USER_AGENT = 'PolicyEngine-Dashboard/1.0';

const allowedOwners = (process.env.GITHUB_ALLOWED_OWNERS || 'PolicyEngine')
  .split(',')
  .map(owner => owner.trim().toLowerCase())
  .filter(Boolean);

function errorResponse(message: string, status: number): NextResponse {
  return NextResponse.json({ error: message }, { status });
}

function isAllowedGitHubPath(path: string): boolean {
  const parts = path.split('/').filter(Boolean);

  if (parts[0] !== 'repos' || parts.length < 4) {
    return false;
  }

  const [, owner, repo, ...rest] = parts;
  if (!allowedOwners.includes(owner.toLowerCase())) {
    return false;
  }

  if (!/^[A-Za-z0-9_.-]+$/.test(owner) || !/^[A-Za-z0-9_.-]+$/.test(repo)) {
    return false;
  }

  if (rest[0] === 'branches' && rest.length === 1) {
    return true;
  }

  if (rest[0] === 'commits' && rest.length === 1) {
    return true;
  }

  if (rest[0] !== 'actions') {
    return false;
  }

  if (rest[1] === 'runs' && rest.length === 2) {
    return true;
  }

  if (
    rest[1] === 'runs' &&
    rest.length === 4 &&
    /^\d+$/.test(rest[2]) &&
    rest[3] === 'artifacts'
  ) {
    return true;
  }

  if (
    rest[1] === 'artifacts' &&
    rest.length === 4 &&
    /^\d+$/.test(rest[2]) &&
    rest[3] === 'zip'
  ) {
    return true;
  }

  return false;
}

export async function GET(request: NextRequest): Promise<NextResponse> {
  const token = process.env.GITHUB_TOKEN || process.env.GITHUB_CONTENTS_READ_TOKEN;
  if (!token) {
    return errorResponse('GitHub token is not configured on the server.', 500);
  }

  const path = request.nextUrl.searchParams.get('path');
  if (!path || !path.startsWith('/') || !isAllowedGitHubPath(path)) {
    return errorResponse('GitHub path is not allowed.', 400);
  }

  const upstreamUrl = new URL(path, GITHUB_API_BASE);
  request.nextUrl.searchParams.forEach((value, key) => {
    if (key !== 'path') {
      upstreamUrl.searchParams.append(key, value);
    }
  });

  const upstream = await fetch(upstreamUrl, {
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: request.headers.get('accept') || 'application/vnd.github.v3+json',
      'User-Agent': USER_AGENT,
    },
  });

  const headers = new Headers();
  const contentType = upstream.headers.get('content-type');
  const contentDisposition = upstream.headers.get('content-disposition');

  if (contentType) {
    headers.set('content-type', contentType);
  }

  if (contentDisposition) {
    headers.set('content-disposition', contentDisposition);
  }

  return new NextResponse(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers,
  });
}
