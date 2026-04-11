/**
 * Model preset metadata served by `GET /api/jobs/presets` (TASKS.md 3.2).
 *
 * The Python `model/presets.py` module is the canonical source of truth
 * for preset hyperparameters and the auto-select heuristic. The Express
 * server doesn't run any ML, so it doesn't need the hyperparameter dicts
 * — it only needs the `{id, label, description}` triples that populate
 * the frontend dropdown. We mirror that subset here as a TypeScript
 * constant rather than spawning Python or maintaining a JSON manifest:
 *
 *   - Spawning Python on every API call adds latency and a process
 *     dependency the Express server otherwise doesn't have.
 *   - A JSON manifest would be a third file to keep in sync.
 *   - The list is short (4 entries) and changes rarely.
 *
 * If you add a preset:
 *   1. Add it to `model/presets.py` (`PRESET_INFO` and
 *      `PRESET_DEFINITIONS`).
 *   2. Add it to `VALID_MODEL_PRESETS` in
 *      `frontend/server/middleware/validation.ts`.
 *   3. Add it to `MODEL_PRESETS` below.
 *   4. The frontend dropdown picks it up automatically.
 *
 * The Python tests in `tests/test_model_presets.py` include a
 * drift-guard that fails if the Python `PRESET_INFO` ids change without
 * also updating this list.
 */

export interface ModelPresetInfo {
  /** Stable preset id sent back to the worker via the `modelPreset` field. */
  id: 'auto' | 'small' | 'medium' | 'large';
  /** Short human-readable label for the dropdown option. */
  label: string;
  /** Longer description shown as a hint under the dropdown. */
  description: string;
}

export const MODEL_PRESETS: ReadonlyArray<ModelPresetInfo> = [
  {
    id: 'auto',
    label: 'Auto',
    description:
      'Pick a preset automatically based on the shape of the uploaded CSV. Recommended.',
  },
  {
    id: 'small',
    label: 'Small',
    description:
      'Compact 1-layer model with a 4-dimensional latent space. Best for small datasets or CSVs with few categorical columns.',
  },
  {
    id: 'medium',
    label: 'Medium',
    description:
      'Balanced 2-layer model with an 8-dimensional latent space. Good default for typical survey CSVs.',
  },
  {
    id: 'large',
    label: 'Large',
    description:
      'Higher-capacity 3-layer model with a 16-dimensional latent space and L2 / batch-norm regularization. Best for wide datasets with many categorical columns.',
  },
];
