# AirfRANS_remeshed — Phase 2 inspection

Data directory: `data/airfrans_remeshed`
Parquet shards: 3; samples: 1000

## Card metadata

- `in_scalars_names`: ['angle_of_attack', 'inlet_velocity']
- `out_scalars_names`: ['C_D', 'C_L']
- `in_fields_names`: ['implicit_distance', 'zone_0', 'zone_1', 'zone_2', 'zone_3', 'zone_4', 'zone_5']
- `out_fields_names`: ['nut', 'Ux', 'Uy', 'p']
- `in_meshes_names`: ['/Base_2_2/Zone']

## Splits (from the card; row indices into `all_samples`)

| split | n | min idx | max idx |
|---|---|---|---|
| `aoa_train` | 804 | 1 | 999 |
| `full_train` | 800 | 0 | 999 |
| `reynolds_train` | 504 | 254 | 757 |
| `scarce_train` | 200 | 2 | 993 |
| `aoa_test` | 196 | 0 | 995 |
| `full_test` | 200 | 9 | 996 |
| `reynolds_test` | 496 | 0 | 999 |
| `ML4PhySim_Challenge_train` | 103 | 255 | 751 |

## First samples

### Sample 0

| scalar | value |
|---|---|
| `C_D` | 0.0106365 |
| `C_L` | -0.321047 |
| `angle_of_attack` | -0.0725359 |
| `inlet_velocity` | 31.283 |

- fields: `Uy`, `p`, `nut`, `implicit_distance`, `Ux`
- nodes: 7418; elements: TRI_3 × 13487
- bases: ['Base_2_2']; zones: ['Zone']
- nodal tags: none
- node bounding box: x ∈ [-2.000, 4.000], y ∈ [-1.500, 1.500]

### Sample 1

| scalar | value |
|---|---|
| `C_D` | 0.0109207 |
| `C_L` | 0.56922 |
| `angle_of_attack` | 0.0626224 |
| `inlet_velocity` | 31.382 |

- fields: `Uy`, `p`, `nut`, `implicit_distance`, `Ux`
- nodes: 7898; elements: TRI_3 × 14586
- bases: ['Base_2_2']; zones: ['Zone']
- nodal tags: none
- node bounding box: x ∈ [-2.000, 4.000], y ∈ [-1.500, 1.500]

### Sample 2

| scalar | value |
|---|---|
| `C_D` | 0.0292549 |
| `C_L` | 1.62036 |
| `angle_of_attack` | 0.239337 |
| `inlet_velocity` | 31.468 |

- fields: `Uy`, `p`, `nut`, `implicit_distance`, `Ux`
- nodes: 7225; elements: TRI_3 × 13706
- bases: ['Base_2_2']; zones: ['Zone']
- nodal tags: none
- node bounding box: x ∈ [-2.000, 4.000], y ∈ [-1.500, 1.500]

## Aggregate over all samples

| quantity | min | median | max |
|---|---|---|---|
| angle_of_attack (raw) | -0.0862193 | 0.0696997 | 0.260508 |
| α (deg) | -4.94 | 3.9935 | 14.926 |
| inlet_velocity (m/s) | 31.283 | 62.404 | 93.592 |
| Re = U∞·c/ν | 2.0185e+06 | 4.02655e+06 | 6.03891e+06 |
| C_L | -0.53363 | 0.683128 | 1.89328 |
| C_D | 0.00690439 | 0.0107423 | 0.0459486 |
| mesh nodes | 4203 | 7738.5 | 9711 |
| wall nodes (boundary loop off the clip box) | 473 | 1115.5 | 1670 |
| max |implicit_distance| on wall nodes | 1.89758e-06 | 5.23963e-06 | 0.000317815 |

ν(T = 298.15 K) = 1.549815e-05 m²/s (AirfRANS polynomial); c = 1.0 m.

Per-sample identifier: the samples carry no original AirfRANS simulation name (no name scalar, no tag, no metadata field); `sample_id` is the row index into `all_samples`, which is also what the card's split lists index.

## Gate 2

- [x] C_L and C_D are per-sample scalars
- [x] angle of attack and inlet velocity present
- [x] AoA units are radians, range ≈ [−0.0873, 0.2618]
- [x] derived Re within ≈ [2e6, 6e6]
- [x] aerofoil wall identifiable (single closed boundary loop off the clip box, ≥ 50 nodes, all with |implicit_distance| < 0.001)
- [x] 1000 samples

**Gate 2: PASS**
