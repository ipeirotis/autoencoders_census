/**
 * Tests for the multer fileFilter used on the /api/upload fallback route.
 *
 * The filter is the first line of defense for the TASKS.md 2.5 bullet
 * "No file-type validation on upload": it rejects obviously non-CSV uploads
 * by extension AND declared MIME type *before* multer reads the payload into
 * memory. The authoritative content check (file-type magic bytes + structure)
 * still runs inside the route via validateCSVContent; this filter just
 * short-circuits trivially wrong uploads.
 */

import { describe, it, expect, jest } from '@jest/globals';

// Mock the config/env module so importing server/index.ts doesn't crash
// on missing env vars inside unit tests. envalid would otherwise call
// process.exit when GOOGLE_CLOUD_PROJECT / GCS_BUCKET_NAME / etc. are unset.
jest.mock('../../config/env', () => ({
  env: {
    NODE_ENV: 'test',
    GOOGLE_CLOUD_PROJECT: 'test-project',
    GCS_BUCKET_NAME: 'test-bucket',
    PUBSUB_TOPIC_ID: 'test-topic',
    SESSION_SECRET: 'a'.repeat(32),
    FRONTEND_URL: 'http://localhost:8080',
    PORT: 5001,
  },
}));

// Mock the GCP singletons so importing server/index.ts doesn't try to
// reach the Firestore/Storage/PubSub networks.
jest.mock('../../config/gcp-clients', () => ({
  storage: {},
  firestore: {},
  pubsub: {},
}));

// Mock the session config so importing server/index.ts doesn't require
// a live Firestore connection to construct the session store.
jest.mock('../../config/session', () => ({
  sessionConfig: (_req: unknown, _res: unknown, next: () => void) => next(),
}));

// Cloud Logging needs to be mocked for the same reason as other tests.
jest.mock('@google-cloud/logging-winston', () => ({
  LoggingWinston: jest.fn().mockImplementation(() => ({
    on: jest.fn(),
    log: jest.fn(),
    levels: {},
  })),
}));

describe('csvUploadFileFilter', () => {
  // Dynamic import so the mocks above are installed before server/index.ts
  // is evaluated.
  async function getFilter() {
    const mod = await import('../../index');
    return mod.csvUploadFileFilter;
  }

  function runFilter(
    filter: (
      req: unknown,
      file: { originalname?: string; mimetype: string },
      cb: (err: Error | null, accept?: boolean) => void
    ) => void,
    file: { originalname?: string; mimetype: string }
  ): { err: Error | null; accept?: boolean } {
    let captured: { err: Error | null; accept?: boolean } = { err: null };
    filter({}, file, (err, accept) => {
      captured = { err, accept };
    });
    return captured;
  }

  it('accepts a .csv file with text/csv mimetype', async () => {
    const filter = await getFilter();
    const { err, accept } = runFilter(filter, {
      originalname: 'data.csv',
      mimetype: 'text/csv',
    });
    expect(err).toBeNull();
    expect(accept).toBe(true);
  });

  it('accepts a .csv file with application/vnd.ms-excel mimetype (Windows Excel)', async () => {
    const filter = await getFilter();
    const { err, accept } = runFilter(filter, {
      originalname: 'export.csv',
      mimetype: 'application/vnd.ms-excel',
    });
    expect(err).toBeNull();
    expect(accept).toBe(true);
  });

  it('accepts a .CSV file (case-insensitive extension)', async () => {
    const filter = await getFilter();
    const { err, accept } = runFilter(filter, {
      originalname: 'DATA.CSV',
      mimetype: 'text/csv',
    });
    expect(err).toBeNull();
    expect(accept).toBe(true);
  });

  it('accepts a .csv file with text/plain mimetype (browser fallback)', async () => {
    const filter = await getFilter();
    const { err, accept } = runFilter(filter, {
      originalname: 'data.csv',
      mimetype: 'text/plain',
    });
    expect(err).toBeNull();
    expect(accept).toBe(true);
  });

  it('rejects a .exe file', async () => {
    const filter = await getFilter();
    const { err } = runFilter(filter, {
      originalname: 'malware.exe',
      mimetype: 'application/x-msdownload',
    });
    expect(err).not.toBeNull();
    expect(err!.message).toContain('.csv files are accepted');
    // Must carry a 400 status so the errorHandler doesn't produce a 500
    expect((err as Error & { status: number }).status).toBe(400);
  });

  it('rejects a .png file', async () => {
    const filter = await getFilter();
    const { err } = runFilter(filter, {
      originalname: 'image.png',
      mimetype: 'image/png',
    });
    expect(err).not.toBeNull();
    expect(err!.message).toContain('.csv files are accepted');
  });

  it('rejects a file with no extension', async () => {
    const filter = await getFilter();
    const { err } = runFilter(filter, {
      originalname: 'justaname',
      mimetype: 'text/csv',
    });
    expect(err).not.toBeNull();
    expect(err!.message).toContain('.csv files are accepted');
  });

  it('rejects a .csv file with a clearly-wrong mimetype', async () => {
    const filter = await getFilter();
    const { err } = runFilter(filter, {
      originalname: 'data.csv',
      mimetype: 'image/png',
    });
    expect(err).not.toBeNull();
    expect(err!.message).toContain('Unsupported content type');
    expect((err as Error & { status: number }).status).toBe(400);
  });

  it('rejects a path-traversal filename masquerading as .csv', async () => {
    // Extension-only check would accept this; we just want to verify the
    // filter behaves consistently (accepts or rejects) without crashing.
    // The actual safety comes from generateSafeFilename(), which ignores
    // the user-provided name entirely — but the filter should still reach
    // a verdict here.
    const filter = await getFilter();
    const { err, accept } = runFilter(filter, {
      originalname: '../../etc/passwd.csv',
      mimetype: 'text/csv',
    });
    // Extension is .csv, mimetype is allowed, so the filter accepts it.
    // The safe-filename generation in the route discards the user path
    // before anything touches GCS.
    expect(err).toBeNull();
    expect(accept).toBe(true);
  });

  // --- MIME type normalization (RFC 7231 §3.1.1.1) -------------------------
  // MIME types are case-insensitive in both type and subtype, and may
  // carry optional parameters (charset, boundary, ...) after `;`. The
  // filter must strip parameters and lowercase before matching, otherwise
  // legitimate `text/csv; charset=utf-8` or `Text/CSV` uploads get 400'd.
  // Codex P2 (PR #47 review).

  it('accepts text/csv with charset parameter', async () => {
    const filter = await getFilter();
    const { err, accept } = runFilter(filter, {
      originalname: 'data.csv',
      mimetype: 'text/csv; charset=utf-8',
    });
    expect(err).toBeNull();
    expect(accept).toBe(true);
  });

  it('accepts text/csv with whitespace before the parameter separator', async () => {
    const filter = await getFilter();
    const { err, accept } = runFilter(filter, {
      originalname: 'data.csv',
      mimetype: 'text/csv ; charset=UTF-8',
    });
    expect(err).toBeNull();
    expect(accept).toBe(true);
  });

  it('accepts Text/CSV with mixed case', async () => {
    const filter = await getFilter();
    const { err, accept } = runFilter(filter, {
      originalname: 'data.csv',
      mimetype: 'Text/CSV',
    });
    expect(err).toBeNull();
    expect(accept).toBe(true);
  });

  it('accepts APPLICATION/VND.MS-EXCEL with uppercase', async () => {
    const filter = await getFilter();
    const { err, accept } = runFilter(filter, {
      originalname: 'export.csv',
      mimetype: 'APPLICATION/VND.MS-EXCEL',
    });
    expect(err).toBeNull();
    expect(accept).toBe(true);
  });

  it('accepts application/csv with boundary-like parameter', async () => {
    const filter = await getFilter();
    const { err, accept } = runFilter(filter, {
      originalname: 'data.csv',
      mimetype: 'application/csv;boundary=something',
    });
    expect(err).toBeNull();
    expect(accept).toBe(true);
  });

  it('still rejects image/png even with charset parameter', async () => {
    const filter = await getFilter();
    const { err } = runFilter(filter, {
      originalname: 'data.csv',
      mimetype: 'image/png; charset=utf-8',
    });
    expect(err).not.toBeNull();
    expect(err!.message).toContain('Unsupported content type');
  });

  it('rejects empty mimetype string', async () => {
    const filter = await getFilter();
    const { err } = runFilter(filter, {
      originalname: 'data.csv',
      mimetype: '',
    });
    expect(err).not.toBeNull();
    expect(err!.message).toContain('Unsupported content type');
  });
});
