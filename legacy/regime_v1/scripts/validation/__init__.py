"""Regime-aware CFD validation utilities for CamberLab.

Submodules:
    parsers                — load NASA TMR / Abbott / Ladson / Coles & Wadcock data
    fetch_reference_data   — re-download every file referenced in metadata.json
    generate_xfoil_reference — produce Regime C reference via XFOIL
    generate_validation_cases — materialize OpenFOAM cases from metadata.json
    compare                — Cl, Cd, Cp comparison and error metrics
    plots                  — comparison and convergence plots
    report                 — per-regime markdown report writer
"""
