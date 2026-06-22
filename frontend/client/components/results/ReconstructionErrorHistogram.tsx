/**
 * Histogram of every row's reconstruction error across the whole dataset.
 *
 * The worker bins the errors server-side (numpy histogram, 40 bins) and writes
 * { binEdges, counts, totalRows, outlierThreshold } onto the job document, so
 * this component just renders pre-computed bars — no charting library needed
 * (recharts is intentionally not a dependency here).
 *
 * Reading it: most rows reconstruct with low error (the tall bars on the left);
 * the flagged outliers are the high-error tail on the right, drawn in red. The
 * outlier threshold is the smallest error among the top-100 flagged rows.
 */

interface ErrorHistogram {
  binEdges: number[];
  counts: number[];
  totalRows: number;
  outlierThreshold?: number;
}

export function ReconstructionErrorHistogram({
  histogram,
}: {
  histogram: ErrorHistogram;
}) {
  const { binEdges, counts, totalRows, outlierThreshold } = histogram;

  // Defensive: numpy histogram guarantees binEdges.length === counts.length + 1.
  if (!counts?.length || !binEdges || binEdges.length < 2) return null;

  const maxCount = Math.max(...counts, 1);
  // Reconstruction-error distributions are heavily right-skewed: the vast
  // majority of rows reconstruct well (one huge bar) and the outliers are a
  // sparse tail. On a linear y-axis the tail is invisible, so we scale bar
  // heights by log(count) to compress the bulk and lift the tail — letting you
  // see the whole shape at once. (X stays linear; the error values are already
  // ~0–1, so log-x wouldn't help.)
  const logMax = Math.log1p(maxCount);
  const fmt = (n: number) => n.toFixed(3);

  return (
    <div className="border rounded-lg p-4">
      <div className="flex items-baseline justify-between mb-1">
        <h3 className="font-semibold text-sm">Reconstruction error distribution</h3>
        <span className="text-xs text-muted-foreground">
          {totalRows.toLocaleString()} rows
        </span>
      </div>
      <p className="text-xs text-muted-foreground mb-3">
        Most rows reconstruct with low error; flagged outliers are the high-error tail
        {outlierThreshold !== undefined ? ` (error ≥ ${fmt(outlierThreshold)})` : ""}.
        Bar heights use a log scale so the sparse outlier tail stays visible.
      </p>

      {/* Bars. Height ∝ log(count) (see logMax above) so a few-thousand-row
          "normal" bin and a handful-of-rows outlier bin are both legible. */}
      <div className="flex items-end gap-px h-40" role="img" aria-label="Histogram of reconstruction errors (log-scaled counts)">
        {counts.map((count, i) => {
          const left = binEdges[i];
          const right = binEdges[i + 1];
          const isOutlierBin =
            outlierThreshold !== undefined && right > outlierThreshold;
          const heightPct =
            count > 0 ? Math.max((Math.log1p(count) / logMax) * 100, 3) : 0;
          return (
            <div
              key={i}
              className={`flex-1 rounded-t ${isOutlierBin ? "bg-red-500" : "bg-blue-500/70"}`}
              style={{ height: `${heightPct}%` }}
              title={`${fmt(left)}–${fmt(right)}: ${count.toLocaleString()} row${count === 1 ? "" : "s"}`}
            />
          );
        })}
      </div>

      <div className="flex justify-between text-[10px] text-muted-foreground mt-1">
        <span>{fmt(binEdges[0])}</span>
        <span>reconstruction error →</span>
        <span>{fmt(binEdges[binEdges.length - 1])}</span>
      </div>

      {outlierThreshold !== undefined && (
        <div className="flex items-center gap-4 mt-3 text-xs">
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-3 rounded-sm bg-blue-500/70" /> normal
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-3 rounded-sm bg-red-500" /> flagged outlier
          </span>
        </div>
      )}
    </div>
  );
}
