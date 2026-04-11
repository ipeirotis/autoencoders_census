"""Tests for model.presets — the preset catalog and auto-select
heuristic introduced in TASKS.md 3.2.

These tests cover:

1. The static catalog (`PRESET_INFO`, `PRESET_DEFINITIONS`,
   `VALID_PRESETS`) and the relationships between them.
2. `normalize_preset_name` — liberal coercion of arbitrary input to
   a valid id.
3. `auto_select_preset` — the breakpoint heuristic for picking a
   concrete preset given the dataset shape.
4. `get_preset_config` — fresh-copy semantics and error handling.
5. `build_model_config` — the high-level entry point used by the
   worker / Vertex training container.
6. `worker.PubSubMessage` — the new optional `modelPreset` field
   accepts both presence and absence without breaking the existing
   schema.
7. `worker._build_vertex_training_args` — the `--model-preset` CLI
   flag is forwarded only when supplied (rolling-deploy contract).
8. The frontend TypeScript preset list mirrors `PRESET_INFO` (drift
   guard).

The tests intentionally avoid loading TensorFlow — `model.presets`
is a pure-Python module with no ML dependencies, and we want
`pytest tests/test_model_presets.py` to be fast.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from model.presets import (
    AUTO_SELECT_MIN_INPUT_DIM_FOR_LARGE,
    AUTO_SELECT_MIN_INPUT_DIM_FOR_MEDIUM,
    AUTO_SELECT_MIN_ROWS_FOR_LARGE,
    AUTO_SELECT_MIN_ROWS_FOR_MEDIUM,
    DEFAULT_PRESET,
    PRESET_AUTO,
    PRESET_DEFINITIONS,
    PRESET_INFO,
    PRESET_LARGE,
    PRESET_MEDIUM,
    PRESET_SMALL,
    VALID_PRESETS,
    auto_select_preset,
    build_model_config,
    get_preset_config,
    list_presets,
    normalize_preset_name,
)


# ---------------------------------------------------------------------------
# Static catalog
# ---------------------------------------------------------------------------

class TestPresetCatalog:
    """Sanity checks on the constants exported by model.presets."""

    def test_default_preset_is_auto(self):
        assert DEFAULT_PRESET == PRESET_AUTO

    def test_valid_presets_includes_all_named_presets_and_auto(self):
        assert set(VALID_PRESETS) == {
            PRESET_AUTO, PRESET_SMALL, PRESET_MEDIUM, PRESET_LARGE,
        }

    def test_preset_definitions_has_concrete_presets_only(self):
        # Auto is a sentinel and must not appear in PRESET_DEFINITIONS —
        # build_model_config catches this and raises a clear error.
        assert PRESET_AUTO not in PRESET_DEFINITIONS
        assert set(PRESET_DEFINITIONS) == {PRESET_SMALL, PRESET_MEDIUM, PRESET_LARGE}

    def test_preset_info_ids_match_valid_presets(self):
        info_ids = {entry["id"] for entry in PRESET_INFO}
        assert info_ids == set(VALID_PRESETS)

    def test_preset_info_entries_have_required_fields(self):
        for entry in PRESET_INFO:
            assert set(entry.keys()) == {"id", "label", "description"}
            assert entry["label"], "label must be non-empty"
            assert entry["description"], "description must be non-empty"

    def test_preset_definitions_have_training_loop_keys(self):
        # The worker pops `epochs` and `batch_size` from the config
        # before passing it to build_autoencoder. Every concrete preset
        # must therefore include both keys.
        for name, config in PRESET_DEFINITIONS.items():
            assert "epochs" in config, f"{name}: missing epochs"
            assert "batch_size" in config, f"{name}: missing batch_size"
            assert isinstance(config["epochs"], int)
            assert isinstance(config["batch_size"], int)
            assert config["epochs"] > 0
            assert config["batch_size"] > 0

    def test_preset_definitions_have_network_shape_keys(self):
        # build_autoencoder uses `latent_space_dim`, `encoder_layers`,
        # and `decoder_layers` directly via config.get(). The presets
        # must populate these or build_autoencoder falls back to its
        # built-in defaults (which were the source of the original
        # "always 2 layers, latent=2" bug).
        required = {
            "latent_space_dim",
            "encoder_layers",
            "decoder_layers",
            "learning_rate",
        }
        for name, config in PRESET_DEFINITIONS.items():
            missing = required - set(config)
            assert not missing, f"{name}: missing required keys {missing}"

    def test_preset_widths_increase_with_size(self):
        # Sanity guard so a future edit cannot accidentally make
        # "large" smaller than "small".
        sml = PRESET_DEFINITIONS[PRESET_SMALL]
        med = PRESET_DEFINITIONS[PRESET_MEDIUM]
        lrg = PRESET_DEFINITIONS[PRESET_LARGE]
        assert sml["latent_space_dim"] < med["latent_space_dim"] < lrg["latent_space_dim"]
        assert sml["encoder_layers"] <= med["encoder_layers"] <= lrg["encoder_layers"]
        assert sml["encoder_units_1"] <= med["encoder_units_1"] <= lrg["encoder_units_1"]


class TestListPresets:
    def test_returns_copy_of_preset_info(self):
        listed = list_presets()
        assert listed == PRESET_INFO

    def test_returned_list_is_independent_from_module_state(self):
        listed = list_presets()
        listed.append({"id": "tampered"})
        # The module-level constant must remain unchanged.
        assert all(p["id"] != "tampered" for p in PRESET_INFO)

    def test_returned_entries_are_independent_dicts(self):
        listed = list_presets()
        listed[0]["label"] = "MUTATED"
        # The module-level entry must remain unchanged.
        assert PRESET_INFO[0]["label"] != "MUTATED"


# ---------------------------------------------------------------------------
# normalize_preset_name
# ---------------------------------------------------------------------------

class TestNormalizePresetName:
    @pytest.mark.parametrize(
        "name", ["auto", "small", "medium", "large"]
    )
    def test_canonical_names_pass_through(self, name):
        assert normalize_preset_name(name) == name

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("AUTO", PRESET_AUTO),
            ("Small", PRESET_SMALL),
            ("MEDIUM", PRESET_MEDIUM),
            ("  Large  ", PRESET_LARGE),
            ("\tlarge\n", PRESET_LARGE),
        ],
    )
    def test_case_and_whitespace_normalization(self, name, expected):
        assert normalize_preset_name(name) == expected

    @pytest.mark.parametrize("name", [None, "", "   "])
    def test_none_and_empty_default_to_auto(self, name):
        assert normalize_preset_name(name) == PRESET_AUTO

    @pytest.mark.parametrize(
        "name", ["unknown", "xxl", "default", "huge", "tiny", "1234"]
    )
    def test_unknown_values_default_to_auto(self, name):
        assert normalize_preset_name(name) == PRESET_AUTO

    def test_non_string_inputs_are_coerced_to_str_then_normalized(self):
        # Numbers, lists, dicts shouldn't crash — they should fall back
        # to auto. The contract is "be liberal in what you accept".
        assert normalize_preset_name(42) == PRESET_AUTO
        assert normalize_preset_name([]) == PRESET_AUTO
        assert normalize_preset_name({}) == PRESET_AUTO


# ---------------------------------------------------------------------------
# auto_select_preset
# ---------------------------------------------------------------------------

class TestAutoSelectPreset:
    def test_thresholds_are_module_constants(self):
        # Spot-check the documented constants so a refactor doesn't
        # silently change behaviour.
        assert AUTO_SELECT_MIN_INPUT_DIM_FOR_MEDIUM == 30
        assert AUTO_SELECT_MIN_ROWS_FOR_MEDIUM == 200
        assert AUTO_SELECT_MIN_INPUT_DIM_FOR_LARGE == 150
        assert AUTO_SELECT_MIN_ROWS_FOR_LARGE == 1000

    @pytest.mark.parametrize(
        "input_dim,n_rows",
        [
            (10, 500),    # narrow
            (29, 5000),   # narrow
            (50, 100),    # too few rows
            (200, 199),   # very few rows
            (5, 5),       # tiny
        ],
    )
    def test_picks_small_for_narrow_or_tiny_data(self, input_dim, n_rows):
        assert auto_select_preset(input_dim, n_rows) == PRESET_SMALL

    @pytest.mark.parametrize(
        "input_dim,n_rows",
        [
            (30, 200),    # exactly at the medium threshold
            (50, 500),    # comfortably medium
            (100, 800),   # still medium
            (149, 5000),  # just below the large threshold
            (200, 999),   # wide but not enough rows for large
        ],
    )
    def test_picks_medium_for_typical_survey_csvs(self, input_dim, n_rows):
        assert auto_select_preset(input_dim, n_rows) == PRESET_MEDIUM

    @pytest.mark.parametrize(
        "input_dim,n_rows",
        [
            (150, 1000),  # exactly at the large threshold
            (300, 10000),
            (500, 50000),
        ],
    )
    def test_picks_large_for_wide_high_volume_data(self, input_dim, n_rows):
        assert auto_select_preset(input_dim, n_rows) == PRESET_LARGE

    def test_zero_rows_picks_small(self):
        # An empty (or near-empty) cleaned dataframe shouldn't crash —
        # we still need to return a valid preset for the worker to use.
        assert auto_select_preset(50, 0) == PRESET_SMALL

    def test_zero_input_dim_picks_small(self):
        assert auto_select_preset(0, 1000) == PRESET_SMALL


# ---------------------------------------------------------------------------
# get_preset_config
# ---------------------------------------------------------------------------

class TestGetPresetConfig:
    @pytest.mark.parametrize(
        "name", [PRESET_SMALL, PRESET_MEDIUM, PRESET_LARGE]
    )
    def test_returns_dict_for_concrete_preset(self, name):
        config = get_preset_config(name)
        assert isinstance(config, dict)
        assert config == PRESET_DEFINITIONS[name]

    @pytest.mark.parametrize(
        "name", [PRESET_SMALL, PRESET_MEDIUM, PRESET_LARGE]
    )
    def test_returns_fresh_copy_each_call(self, name):
        a = get_preset_config(name)
        b = get_preset_config(name)
        assert a is not b
        a["learning_rate"] = 0.999
        # Mutating the copy must not bleed back into the module state.
        assert PRESET_DEFINITIONS[name]["learning_rate"] != 0.999
        # And a fresh call still sees the original value.
        c = get_preset_config(name)
        assert c["learning_rate"] != 0.999

    def test_auto_sentinel_is_rejected(self):
        with pytest.raises(ValueError, match="sentinel"):
            get_preset_config(PRESET_AUTO)

    def test_unknown_name_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset_config("xxl")


# ---------------------------------------------------------------------------
# build_model_config
# ---------------------------------------------------------------------------

class TestBuildModelConfig:
    def test_named_preset_passes_through_unchanged(self):
        config, name = build_model_config(PRESET_MEDIUM, input_dim=999, n_rows=1)
        assert name == PRESET_MEDIUM
        assert config == PRESET_DEFINITIONS[PRESET_MEDIUM]

    def test_auto_resolves_to_small_for_narrow_data(self):
        config, name = build_model_config(PRESET_AUTO, input_dim=10, n_rows=500)
        assert name == PRESET_SMALL
        assert config == PRESET_DEFINITIONS[PRESET_SMALL]

    def test_auto_resolves_to_medium_for_typical_data(self):
        config, name = build_model_config(PRESET_AUTO, input_dim=80, n_rows=2000)
        assert name == PRESET_MEDIUM
        assert config == PRESET_DEFINITIONS[PRESET_MEDIUM]

    def test_auto_resolves_to_large_for_wide_high_volume_data(self):
        config, name = build_model_config(PRESET_AUTO, input_dim=300, n_rows=5000)
        assert name == PRESET_LARGE
        assert config == PRESET_DEFINITIONS[PRESET_LARGE]

    def test_none_falls_back_to_auto(self):
        config, name = build_model_config(None, input_dim=80, n_rows=2000)
        assert name == PRESET_MEDIUM
        assert config == PRESET_DEFINITIONS[PRESET_MEDIUM]

    def test_unknown_string_falls_back_to_auto(self):
        config, name = build_model_config("xxl", input_dim=80, n_rows=2000)
        assert name == PRESET_MEDIUM
        assert config == PRESET_DEFINITIONS[PRESET_MEDIUM]

    def test_returned_config_is_a_copy(self):
        config, _ = build_model_config(PRESET_MEDIUM, input_dim=80, n_rows=2000)
        config["learning_rate"] = 0.999
        # Mutating the returned dict must not bleed back into the module
        # state — the worker pops `epochs`/`batch_size` out of the dict
        # before passing it to build_autoencoder, so a shared reference
        # would corrupt the next call.
        assert PRESET_DEFINITIONS[PRESET_MEDIUM]["learning_rate"] != 0.999

    def test_resolved_name_is_concrete_for_auto(self):
        # The whole point of returning the resolved name is so the
        # worker can persist a concrete (small/medium/large) value on
        # the job document — never 'auto'.
        for input_dim, n_rows in [(10, 500), (80, 2000), (300, 5000)]:
            _, name = build_model_config(PRESET_AUTO, input_dim, n_rows)
            assert name in {PRESET_SMALL, PRESET_MEDIUM, PRESET_LARGE}
            assert name != PRESET_AUTO


# ---------------------------------------------------------------------------
# worker.PubSubMessage — modelPreset is optional and tolerant
# ---------------------------------------------------------------------------

class TestWorkerPubSubMessageWithPreset:
    def test_message_without_preset_is_valid(self):
        # Back-compat: a legacy /start-job request that omits the new
        # field must still parse cleanly. validate_message returns a
        # PubSubMessage with modelPreset == None.
        from worker import validate_message

        msg = validate_message({
            "jobId": "abc",
            "bucket": "test-bucket",
            "file": "test.csv",
        })
        assert msg.modelPreset is None

    def test_message_with_valid_preset_is_accepted(self):
        from worker import validate_message

        msg = validate_message({
            "jobId": "abc",
            "bucket": "test-bucket",
            "file": "test.csv",
            "modelPreset": "medium",
        })
        assert msg.modelPreset == "medium"

    def test_message_with_unknown_preset_value_is_still_accepted(self):
        # The worker is intentionally lenient at the schema layer — it
        # is the normalizer's job to coerce unknown values to 'auto'.
        # We assert that here: PubSubMessage accepts the field, and
        # normalize_preset_name then drops it.
        from worker import validate_message

        msg = validate_message({
            "jobId": "abc",
            "bucket": "test-bucket",
            "file": "test.csv",
            "modelPreset": "xxl",
        })
        assert msg.modelPreset == "xxl"
        assert normalize_preset_name(msg.modelPreset) == PRESET_AUTO


# ---------------------------------------------------------------------------
# worker._build_vertex_training_args — preset CLI flag forwarding
# ---------------------------------------------------------------------------

class TestBuildVertexTrainingArgsWithPreset:
    def test_preset_flag_omitted_when_none(self):
        # Rolling-deploy contract: when the dispatcher does not have a
        # preset to forward, the arg list must NOT include
        # --model-preset so an older trainer:v1 image (which does not
        # yet recognise the flag) parses cleanly.
        from worker import _build_vertex_training_args

        args = _build_vertex_training_args("job1", "bucket1", "file.csv")
        assert all(not a.startswith("--model-preset") for a in args)

    def test_preset_flag_included_when_supplied(self):
        from worker import _build_vertex_training_args

        args = _build_vertex_training_args(
            "job1", "bucket1", "file.csv", model_preset="large"
        )
        assert "--model-preset=large" in args

    def test_max_unique_values_and_preset_can_coexist(self):
        from worker import _build_vertex_training_args

        args = _build_vertex_training_args(
            "job1",
            "bucket1",
            "file.csv",
            max_unique_values=15,
            model_preset="medium",
        )
        assert "--max-unique-values=15" in args
        assert "--model-preset=medium" in args

    @pytest.mark.parametrize("value", ["auto", "AUTO", " Auto ", "\tauto\n"])
    def test_auto_is_omitted_for_rolling_deploy_compat(self, value):
        """Codex P1 r#50: `model_preset='auto'` MUST NOT be forwarded
        as `--model-preset=auto` because an older trainer image that
        pre-dates the flag would exit on the unrecognized CLI arg.
        The trainer treats an absent flag as auto by default (via
        `build_model_config(None, ...)` → `normalize_preset_name`
        → `auto_select_preset`), so omitting the flag is semantically
        identical to passing 'auto' — but only the "omit" variant
        survives a mixed-version rolling deploy.
        """
        from worker import _build_vertex_training_args

        args = _build_vertex_training_args(
            "job1", "bucket1", "file.csv", model_preset=value
        )
        assert all(not a.startswith("--model-preset") for a in args), (
            f"auto sentinel {value!r} must not appear as --model-preset "
            f"(got args={args})"
        )

    @pytest.mark.parametrize("value", ["small", "medium", "large"])
    def test_concrete_presets_are_still_forwarded(self, value):
        # The auto-omit logic must not accidentally drop concrete
        # preset ids: small/medium/large still need to reach the
        # trainer.
        from worker import _build_vertex_training_args

        args = _build_vertex_training_args(
            "job1", "bucket1", "file.csv", model_preset=value
        )
        assert f"--model-preset={value}" in args


# ---------------------------------------------------------------------------
# Drift guard: TypeScript preset list mirrors Python PRESET_INFO
# ---------------------------------------------------------------------------

class TestFrontendPresetMirror:
    """The frontend dropdown reads from a hardcoded TypeScript list in
    ``frontend/server/utils/modelPresets.ts``. That list MUST stay in
    sync with the Python ``PRESET_INFO`` constant — if it doesn't, the
    UI will offer presets the worker rejects (or vice versa).

    Validate the contract by parsing the TS file's id literals and
    comparing them to the Python ids. We deliberately use a coarse
    regex parser instead of a full TS parser because the file shape
    is fixed and the test should not require a Node toolchain to run.
    """

    def test_typescript_preset_ids_match_python(self):
        ts_path = (
            Path(__file__).parent.parent
            / "frontend"
            / "server"
            / "utils"
            / "modelPresets.ts"
        )
        if not ts_path.exists():
            pytest.skip(f"frontend preset list missing at {ts_path}")
        text = ts_path.read_text(encoding="utf-8")
        # Match the inline `id: 'something',` literals inside MODEL_PRESETS.
        ids = set(re.findall(r"id:\s*'([^']+)'", text))
        python_ids = {entry["id"] for entry in PRESET_INFO}
        assert ids == python_ids, (
            f"frontend modelPresets.ts ids {ids} drifted from Python "
            f"PRESET_INFO {python_ids}"
        )

    def test_typescript_validation_allowlist_matches_python(self):
        ts_path = (
            Path(__file__).parent.parent
            / "frontend"
            / "server"
            / "middleware"
            / "validation.ts"
        )
        if not ts_path.exists():
            pytest.skip(f"validation middleware missing at {ts_path}")
        text = ts_path.read_text(encoding="utf-8")
        # Locate VALID_MODEL_PRESETS and extract the string literals.
        match = re.search(
            r"VALID_MODEL_PRESETS\s*=\s*\[([^\]]+)\]",
            text,
        )
        assert match, "Could not find VALID_MODEL_PRESETS in validation.ts"
        ids = set(re.findall(r"'([^']+)'", match.group(1)))
        assert ids == set(VALID_PRESETS), (
            f"frontend validation.ts allowlist {ids} drifted from Python "
            f"VALID_PRESETS {set(VALID_PRESETS)}"
        )
