"""
scripts.surrogate — AirfRANS coefficient surrogate.

Modules
-------
data       : dataset + QA flags, feature schema, frozen task splits
models     : the four model families (GP, RF, MLP, KRG): build / fit / predict
inference  : predict(alpha_deg, Re, naca, family, task) against persisted models
"""
