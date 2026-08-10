# Revision experiments (Aug 2026)

Scripts backing the MISQ-revision analyses. Run from repo root:
`PYTHONPATH=$PWD venv_ae/bin/python revision_experiments/<script>`

- `favorable.py` — native-orientation (label-free) detection leaderboard + directional-reliability (anti-detection) rates across all 9 datasets.
- `theory_verify.py` / `theory_all.py` — empirical verification of the measurement model: single-factor fit + latent-validity estimate (Corr(M,A)) on the multi-check datasets, and the Corollary-1 lower bound on all datasets.
- `fig_mm.py` — builds figures/measurement_model_verification.pdf.
- `expE_robinson.py` — Experiment E on Robinson-Cimpian/SADC: LGBQ-heterosexual drug-disparity change from removing mischievous / AE / CL flags + flag-rate-by-group diagnostic.
- `expE_strength.py` — Experiment E strength sweep: pennycook discernment, alvarez attitude constraint, ogrady MFQ-behavior, effect vs fraction removed (AE/CL/random/AC-failers).
