export interface CSVParseResult {
  rows: Record<string, unknown>[];
  headers: string[];
  totalRows: number;
}

// Reads file, parses CSV content, and returns preview data
export async function parseCSVFile(
  file: File,
  skipRows: number = 0
): Promise<CSVParseResult> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = () => {
      try {
        const content = reader.result as string;
        const lines = content
          .split("\n")
          .map((line) => line.trim())
          .filter((line) => line.length > 0);

        if (lines.length === 0) {
          reject(new Error("CSV file is empty"));
          return;
        }

        // Skip the specified number of rows
        const skippedLines = lines.slice(skipRows);

        if (skippedLines.length === 0) {
          reject(new Error("No rows left after skipping"));
          return;
        }

        const headers = parseCSVLine(skippedLines[0]);
        if (headers.length === 0) {
          reject(new Error("CSV file has no headers"));
          return;
        }

        const rows: Record<string, unknown>[] = [];
        for (let i = 1; i < skippedLines.length; i++) {
          const values = parseCSVLine(skippedLines[i]);
          const row: Record<string, unknown> = {};

          headers.forEach((header, index) => {
            row[header] = values[index] || "";
          });

          rows.push(row);
        }

        if (rows.length === 0) {
          reject(new Error("CSV file has no data rows (only headers)"));
          return;
        }

        resolve({
          rows: rows.slice(0, 20),
          headers,
          totalRows: rows.length,
        });
      } catch (error) {
        reject(
          error instanceof Error
            ? error
            : new Error("Failed to parse CSV file")
        );
      }
    };

    reader.onerror = () => {
      reject(new Error("Failed to read file"));
    };

    reader.readAsText(file);
  });
}

function parseCSVLine(line: string): string[] {
  const result: string[] = [];
  let current = "";
  let insideQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    const nextChar = line[i + 1];

    if (char === '"') {
      if (insideQuotes && nextChar === '"') {
        current += '"';
        i++;
      } else {
        insideQuotes = !insideQuotes;
      }
    } else if (char === "," && !insideQuotes) {
      result.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }

  result.push(current.trim());
  return result;
}
