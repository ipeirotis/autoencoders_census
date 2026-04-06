import { useState, useEffect } from 'react';

interface JobMetadataProps {
  startTime: Date;
  estimatedDuration?: number; // milliseconds
  fileName: string;
  fileSize: number;
}

/**
 * Job metadata component displaying elapsed time, estimated remaining time, file info.
 *
 * Features:
 * - Elapsed time updates every second
 * - Estimated remaining time calculated from estimatedDuration - elapsed
 * - File size formatted in B/KB/MB
 *
 * Per user decision in CONTEXT.md: Display elapsed time, estimated remaining, file name & size.
 */
export function JobMetadata({ startTime, estimatedDuration, fileName, fileSize }: JobMetadataProps) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setElapsed(Date.now() - startTime.getTime());
    }, 1000);
    return () => clearInterval(interval);
  }, [startTime]);

  const formatDuration = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const remaining = estimatedDuration ? estimatedDuration - elapsed : null;

  return (
    <div className="text-sm text-gray-600 space-y-1">
      <p>Elapsed: {formatDuration(elapsed)}</p>
      {remaining && remaining > 0 && (
        <p>Estimated remaining: ~{formatDuration(remaining)}</p>
      )}
      <p>File: {fileName} ({formatFileSize(fileSize)})</p>
    </div>
  );
}
