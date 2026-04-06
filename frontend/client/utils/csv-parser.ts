import Papa from 'papaparse';

export interface CSVParseResult {
  rows: Array<Record<string, any>>;
  headers: string[];
  totalRows: number;
}

/**
 * Parse CSV file using Papa Parse streaming to prevent memory crashes on large files.
 * Uses web worker to avoid blocking UI thread.
 * Previews only first 100 rows to prevent loading entire 50MB+ file into memory.
 * Returns only first 20 rows for display.
 */
export async function parseCSVFile(file: File): Promise<CSVParseResult> {
  return new Promise((resolve, reject) => {
    const rows: Array<Record<string, any>> = [];

    Papa.parse(file, {
      worker: true,        // Use web worker to avoid blocking UI
      header: true,        // First row is headers
      skipEmptyLines: true,
      preview: 100,        // Only parse first 100 rows for preview (prevents memory crash)
      step: (results) => {
        // Collect each parsed row
        rows.push(results.data);
      },
      complete: (results) => {
        // Handle completion
        if (rows.length === 0) {
          reject(new Error('CSV file has no data rows'));
          return;
        }

        // Extract headers from first row's keys
        const headers = Object.keys(rows[0] || {});

        if (headers.length === 0) {
          reject(new Error('CSV file has no headers'));
          return;
        }

        resolve({
          rows: rows.slice(0, 20),  // Show first 20 rows in preview
          headers: headers,
          totalRows: rows.length
        });
      },
      error: (error) => {
        reject(new Error(`CSV parsing failed: ${error.message}`));
      }
    });
  });
}
