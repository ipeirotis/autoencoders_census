"""Model preset configurations for the upload pipeline (TASKS.md 3.2).

Provides three named presets — ``small``, ``medium``, ``large`` — plus an
``auto`` selector that picks one based on the post-vectorization input
dimension and row count of the uploaded CSV. Each preset is a dict of
hyperparameters consumed by :class:`model.autoencoder.AutoencoderModel`'s
``build_autoencoder()``, plus training-loop knobs (``epochs``,
``batch_size``).

Why presets exist
-----------------
Until TASKS.md 3.2 the upload pipeline used a single hardcoded model
config in both ``worker.process_upload_local`` and ``train/task.py``,
with widths derived purely from ``input_dim`` (latent_space_dim =
input_dim * 0.1, encoder_units_1 = input_dim * 0.5). That formula
produced surprising shapes on edge cases (e.g. an input_dim of 4 set
latent_space_dim to 2 and encoder_units_1 to 2 — too narrow to learn
anything useful). The presets give us:

1. Sensible defaults that don't degenerate on tiny inputs.
2. A way to expose user-facing choice in the web UI without hand-rolling
   advanced hyperparameter knobs.
3. A single source of truth that the local worker, the Vertex AI
   training container, and the frontend dropdown all read from.

The presets are deliberately conservative — the goal is "reasonable
result on a typical survey CSV without manual tuning" rather than
"best possible model on every dataset". Hyperparameter search via
:func:`main.search_hyperparameters` remains the path for users who want
optimal configs for a specific dataset.

Public API
----------
- :data:`PRESET_INFO` — list of ``{id, label, description}`` dicts that
  the API endpoint ``GET /api/jobs/presets`` returns to the frontend.
- :func:`list_presets` — return a fresh copy of ``PRESET_INFO``.
- :func:`normalize_preset_name` — coerce arbitrary input to a valid
  preset id, defaulting to ``auto``.
- :func:`auto_select_preset` — pick a concrete preset id given the
  dataset shape.
- :func:`get_preset_config` — return the hyperparameter dict for a
  named (non-auto) preset.
- :func:`build_model_config` — high-level entry point used by
  ``worker.py`` and ``train/task.py``: takes the requested preset name
  (which may be ``auto`` or ``None``) plus the dataset shape, and
  returns ``(config_dict, resolved_name)``.
"""

from __future__ import annotations

from typing import Any

PRESET_AUTO = "auto"
PRESET_SMALL = "small"
PRESET_MEDIUM = "medium"
PRESET_LARGE = "large"

#: Tuple of every preset name the API will accept (including the ``auto``
#: sentinel). Used by the worker / API server to validate the inbound
#: ``modelPreset`` field.
VALID_PRESETS: tuple[str, ...] = (
    PRESET_AUTO,
    PRESET_SMALL,
    PRESET_MEDIUM,
    PRESET_LARGE,
)

#: Default preset when the caller doesn't specify one. ``auto`` picks a
#: concrete preset at runtime based on the shape of the cleaned input.
DEFAULT_PRESET = PRESET_AUTO

#: Frontend-facing preset metadata. The shape mirrors what the
#: ``GET /api/jobs/presets`` endpoint returns. Keep this list in sync
#: with :data:`PRESET_DEFINITIONS` below — the validation in
#: :func:`normalize_preset_name` only accepts ids that appear in
#: :data:`VALID_PRESETS`.
PRESET_INFO: list[dict[str, str]] = [
    {
        "id": PRESET_AUTO,
        "label": "Auto",
        "description": (
            "Pick a preset automatically based on the shape of the "
            "uploaded CSV. Recommended."
        ),
    },
    {
        "id": PRESET_SMALL,
        "label": "Small",
        "description": (
            "Compact 1-layer model with a 4-dimensional latent space. "
            "Best for small datasets or CSVs with few categorical columns."
        ),
    },
    {
        "id": PRESET_MEDIUM,
        "label": "Medium",
        "description": (
            "Balanced 2-layer model with an 8-dimensional latent space. "
            "Good default for typical survey CSVs."
        ),
    },
    {
        "id": PRESET_LARGE,
        "label": "Large",
        "description": (
            "Higher-capacity 3-layer model with a 16-dimensional latent "
            "space and L2 / batch-norm regularization. Best for wide "
            "datasets with many categorical columns."
        ),
    },
]

#: Concrete hyperparameter dicts. Each dict is consumed by
#: :func:`model.autoencoder.AutoencoderModel.build_autoencoder` (which
#: reads keys via ``config.get(...)`` and falls back to its own defaults
#: for anything missing) plus the worker's training loop, which pops
#: the ``epochs`` and ``batch_size`` keys.
PRESET_DEFINITIONS: dict[str, dict[str, Any]] = {
    PRESET_SMALL: {
        "learning_rate": 1e-3,
        "latent_space_dim": 4,
        "latent_activation": "relu",
        "encoder_layers": 1,
        "decoder_layers": 1,
        "encoder_units_1": 32,
        "decoder_units_1": 32,
        "encoder_activation_1": "relu",
        "decoder_activation_1": "relu",
        "encoder_dropout_1": 0.0,
        "decoder_dropout_1": 0.0,
        "encoder_l2_1": 0.0,
        "decoder_l2_1": 0.0,
        "encoder_batch_norm_1": False,
        "decoder_batch_norm_1": False,
        "epochs": 10,
        "batch_size": 32,
    },
    PRESET_MEDIUM: {
        "learning_rate": 1e-3,
        "latent_space_dim": 8,
        "latent_activation": "relu",
        "encoder_layers": 2,
        "decoder_layers": 2,
        "encoder_units_1": 128,
        "encoder_units_2": 64,
        "decoder_units_1": 64,
        "decoder_units_2": 128,
        "encoder_activation_1": "relu",
        "encoder_activation_2": "relu",
        "decoder_activation_1": "relu",
        "decoder_activation_2": "relu",
        "encoder_dropout_1": 0.1,
        "encoder_dropout_2": 0.1,
        "decoder_dropout_1": 0.1,
        "decoder_dropout_2": 0.1,
        "encoder_l2_1": 0.0,
        "encoder_l2_2": 0.0,
        "decoder_l2_1": 0.0,
        "decoder_l2_2": 0.0,
        "encoder_batch_norm_1": False,
        "encoder_batch_norm_2": False,
        "decoder_batch_norm_1": False,
        "decoder_batch_norm_2": False,
        "epochs": 15,
        "batch_size": 32,
    },
    PRESET_LARGE: {
        "learning_rate": 5e-4,
        "latent_space_dim": 16,
        "latent_activation": "relu",
        "encoder_layers": 3,
        "decoder_layers": 3,
        "encoder_units_1": 256,
        "encoder_units_2": 128,
        "encoder_units_3": 64,
        "decoder_units_1": 64,
        "decoder_units_2": 128,
        "decoder_units_3": 256,
        "encoder_activation_1": "relu",
        "encoder_activation_2": "relu",
        "encoder_activation_3": "relu",
        "decoder_activation_1": "relu",
        "decoder_activation_2": "relu",
        "decoder_activation_3": "relu",
        "encoder_dropout_1": 0.1,
        "encoder_dropout_2": 0.1,
        "encoder_dropout_3": 0.1,
        "decoder_dropout_1": 0.1,
        "decoder_dropout_2": 0.1,
        "decoder_dropout_3": 0.1,
        "encoder_l2_1": 1e-4,
        "encoder_l2_2": 1e-4,
        "encoder_l2_3": 1e-4,
        "decoder_l2_1": 1e-4,
        "decoder_l2_2": 1e-4,
        "decoder_l2_3": 1e-4,
        "encoder_batch_norm_1": True,
        "encoder_batch_norm_2": True,
        "encoder_batch_norm_3": True,
        "decoder_batch_norm_1": True,
        "decoder_batch_norm_2": True,
        "decoder_batch_norm_3": True,
        "epochs": 25,
        "batch_size": 64,
    },
}

# Auto-select thresholds. Pulled out as module-level constants so tests
# can reference them without duplicating the magic numbers and so an
# operator can monkey-patch them in a one-off script if a particular
# dataset family needs different breakpoints.
AUTO_SELECT_MIN_ROWS_FOR_MEDIUM = 200
AUTO_SELECT_MIN_INPUT_DIM_FOR_MEDIUM = 30
AUTO_SELECT_MIN_INPUT_DIM_FOR_LARGE = 150
AUTO_SELECT_MIN_ROWS_FOR_LARGE = 1000


def list_presets() -> list[dict[str, str]]:
    """Return a fresh deep copy of :data:`PRESET_INFO`.

    The API endpoint serializes this directly to JSON for the frontend
    dropdown. Returning a copy keeps callers from mutating the
    module-level metadata.
    """
    return [dict(entry) for entry in PRESET_INFO]


def normalize_preset_name(name: Any) -> str:
    """Coerce an arbitrary preset name to a valid id.

    Args:
        name: The requested preset id. Accepts strings (case-insensitive,
            stripped of surrounding whitespace), ``None``, or any other
            type — non-string inputs are silently coerced to ``str()``
            before validation. Unknown / empty values fall back to
            :data:`DEFAULT_PRESET` (``auto``).

    Returns:
        A valid id from :data:`VALID_PRESETS`. Never raises — the
        contract is "be liberal in what you accept" so a stale frontend
        cannot crash the worker by sending an unrecognized preset
        name. The worker logs the resolved value alongside the
        originally-requested value so operators can spot drift.
    """
    if name is None:
        return DEFAULT_PRESET
    candidate = str(name).strip().lower()
    if candidate in VALID_PRESETS:
        return candidate
    return DEFAULT_PRESET


def auto_select_preset(input_dim: int, n_rows: int) -> str:
    """Pick a concrete preset id given the cleaned dataset shape.

    Args:
        input_dim: Number of columns in the post-vectorization one-hot
            matrix. This is the dimension the autoencoder actually
            ingests, not the raw CSV column count.
        n_rows: Number of rows in the cleaned dataframe (after the
            Rule-of-N filter).

    Returns:
        One of :data:`PRESET_SMALL`, :data:`PRESET_MEDIUM`,
        :data:`PRESET_LARGE`.

    Heuristic:
        - **small** when ``input_dim < 30`` or ``n_rows < 200``. Tiny
          datasets cannot support a wider model without overfitting,
          and a 1-layer encoder with 4 latent dims is plenty for the
          handful of free parameters they imply.
        - **large** when ``input_dim >= 150`` *and* ``n_rows >= 1000``.
          Wide inputs benefit from the extra encoder depth and the
          larger latent space, but only if there is enough training
          data to fit ~50K parameters without memorising.
        - **medium** otherwise — the bulk of typical survey CSVs.

    The thresholds are intentionally coarse: this is a UX guard rail,
    not a hyperparameter optimizer. Users who want fine-grained control
    can pick the preset explicitly via the dropdown.
    """
    if input_dim < AUTO_SELECT_MIN_INPUT_DIM_FOR_MEDIUM:
        return PRESET_SMALL
    if n_rows < AUTO_SELECT_MIN_ROWS_FOR_MEDIUM:
        return PRESET_SMALL
    if (
        input_dim >= AUTO_SELECT_MIN_INPUT_DIM_FOR_LARGE
        and n_rows >= AUTO_SELECT_MIN_ROWS_FOR_LARGE
    ):
        return PRESET_LARGE
    return PRESET_MEDIUM


def get_preset_config(preset_name: str) -> dict[str, Any]:
    """Return a fresh copy of the named preset's hyperparameter dict.

    Args:
        preset_name: A concrete preset id (``small`` / ``medium`` /
            ``large``). Passing ``auto`` is a programming error — call
            :func:`auto_select_preset` first to resolve it.

    Raises:
        ValueError: If the name is ``auto`` (sentinel, must be
            resolved first) or not in :data:`PRESET_DEFINITIONS`.
    """
    if preset_name == PRESET_AUTO:
        raise ValueError(
            "PRESET_AUTO is a sentinel; resolve it via auto_select_preset() "
            "before calling get_preset_config()."
        )
    if preset_name not in PRESET_DEFINITIONS:
        valid = ", ".join(sorted(PRESET_DEFINITIONS))
        raise ValueError(
            f"Unknown preset {preset_name!r}; valid options are: {valid}"
        )
    return dict(PRESET_DEFINITIONS[preset_name])


def build_model_config(
    preset_name: Any,
    input_dim: int,
    n_rows: int,
) -> tuple[dict[str, Any], str]:
    """Resolve a (possibly auto / possibly invalid) preset name into a
    concrete hyperparameter dict.

    This is the high-level entry point used by ``worker.py`` and
    ``train/task.py``. It folds together the steps a caller would
    otherwise have to chain by hand:

    1. ``normalize_preset_name`` — coerce arbitrary input to a valid id.
    2. ``auto_select_preset`` — if the caller asked for ``auto``,
       pick a concrete preset based on the dataset shape.
    3. ``get_preset_config`` — return the hyperparameter dict.

    Args:
        preset_name: The requested preset id. Anything :func:`normalize_preset_name`
            doesn't recognise (including ``None``) is treated as ``auto``.
        input_dim: Post-vectorization input width.
        n_rows: Number of rows in the cleaned dataframe.

    Returns:
        ``(config_dict, resolved_name)`` — the resolved name is the
        concrete preset id (``small`` / ``medium`` / ``large``), which
        the worker writes to the job document so the frontend can
        display "model: medium" alongside the results.
    """
    name = normalize_preset_name(preset_name)
    if name == PRESET_AUTO:
        name = auto_select_preset(input_dim, n_rows)
    config = get_preset_config(name)
    return config, name
