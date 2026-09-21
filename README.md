# UFO-MGen

**Topology-Stratified Materials Discovery with A Flow-Based Generative Model**

UFO-MGen is a crystal generative framework built around a symmetry-aware
Wyckoff representation. This repository contains the processed dataset,
representation and model implementations, generation/screening/evaluation
utilities, benchmark configurations, and representative DFT-validated
structures used in the manuscript.

Pretrained UFO-MGen checkpoints are distributed separately from GitHub;
checkpoint inventories and SHA256 hashes are provided under
[`manifests/`](manifests/).

## Installation

Python >= 3.10 is required.

```bash
git clone https://github.com/jingyizhou0112-hash/UFO-MGen-pre.git
cd UFO-MGen-pre
pip install -e .
```

Optional dependencies:

```bash
pip install -e ".[screening]"
pip install -e ".[mlip]"
pip install -e ".[mech]"
```

The MLIP extra provides interfaces to CHGNet, MACE, MatterSim, and phonopy.
External pretrained MLIP weights are not redistributed by this repository.

## Wyckoff representation

A crystal is represented by its space group, occupied Wyckoff orbits, one
species per orbit, free Wyckoff coordinates, and independent lattice degrees
of freedom.

```text
D_rep = D_L + sum_k d_k
rho   = D_rep / (3N + 6)
```

Here `D_L` is the number of independent lattice parameters and `d_k` is
the number of free coordinates of Wyckoff orbit `k`.

```python
from ufo_mgen.wyckoff import encode_cif

enc = encode_cif("structure.cif")
print(enc.spacegroup)
print(enc.scaffold)
print(enc.orbit_species)
print(enc.D_rep, enc.rho)
```

Examples:

```bash
python examples/encode_structure.py
python examples/roundtrip.py
python examples/explore_dataset.py
```

The serialized decode vector stores six lattice values plus the free Wyckoff
coordinates. `D_rep` is the intrinsic number of independent degrees of
freedom, not the serialized vector length.

## Model

| Stage | Module | Role |
|---|---|---|
| I | **HTS** - Hierarchical Topology Selection | composition -> space group -> orbit count -> Wyckoff configuration |
| II | **COM** - Chemical Occupancy Module | Wyckoff scaffold -> species assignment by orbit |
| III | **SWG** - Structured Wyckoff-aware Generation | scaffold + chemistry -> lattice and free Wyckoff coordinates |

For the released production profile, the Stage-I topology interface is
materialized through
[`data/route_inventory_124.csv`](data/route_inventory_124.csv). The manifest
provides the space-group/scaffold targets and their Stage-III checkpoint
mapping while preserving the Stage-I topology provenance.

Stage II is a chemistry-aware autoregressive Transformer. Stage III uses a
flow-matching vector field for the lattice and free Wyckoff coordinates, with
periodic wrapping of the coordinate flow on `[0, 1)`.

```bash
python examples/inspect_model.py
```

## Dataset

The released dataset is derived from Materials Project crystal structures and
processed into the UFO-MGen Wyckoff representation used for training and
generation. The processed files are provided under [`data/`](data/).

Other crystal datasets can be used for training once represented in the same
Wyckoff format. CIF structures can be converted with the provided
[`examples/encode_structure.py`](examples/encode_structure.py) example or the
[`ufo_mgen.wyckoff`](src/ufo_mgen/wyckoff/) encoding utilities.

## Configuration

Two public YAML configurations are provided:

| Config | Purpose |
|---|---|
| `configs/ufo_mgen_final.yaml` | main UFO-MGen model and route-based generation profile |
| `configs/mp20.yaml` | MP20 benchmark and extrapolation profile |

The main configuration records dataset paths, model architecture, checkpoint
locations, sampling parameters, and symmetry tolerances.
`generation.candidates_per_route` is intentionally `null`; users set the
desired sample count explicitly.

```bash
python -m ufo_mgen.generation.generate_main \
    --config configs/ufo_mgen_final.yaml \
    --weights-dir weights/ \
    --num-samples 100 \
    --output generated/
```

The released route inventory contains 124 route rows, 119 distinct
space-group/scaffold targets, and 120 Stage-III checkpoint files.

## Post-generation screening

[`ufo_mgen.screening`](src/ufo_mgen/screening/) provides candidate checks for
scaffold consistency, geometry, chemistry, novelty/deduplication, MLFF risk,
relaxation, and post-relaxation filtering.

```bash
python examples/screening_demo.py
```

The quick formation-energy prescreen is optional and disabled by default.
Screening thresholds are collected in
[`src/ufo_mgen/screening/thresholds.py`](src/ufo_mgen/screening/thresholds.py).

## Multi-stability evaluation (MSE)

The MSE implementation evaluates thermodynamic, lattice-dynamical, and
finite-temperature stability using three universal MLIPs:

- CHGNet
- MACE
- MatterSim

A candidate is selected when at least two of the three MLIP backends classify
it as stable under the MSE criteria.

> **Recommended for follow-up studies.** The MSE/MLIP utilities are designed as
> a reusable high-throughput validation layer for newly generated crystals.
> Rather than building separate CHGNet, MACE, and MatterSim workflows from
> scratch, `ufo_mgen.mlip` provides a common harness for calculator loading,
> structural relaxation, formation-energy evaluation, phonon calculations, and
> finite-temperature MD. The resulting quantities can then be evaluated with
> the same MSE gates in `ufo_mgen.evaluation.mse`. This substantially reduces
> the workflow engineering needed to screen a new generated set with multiple
> MLIPs and makes the MSE protocol easy to reuse beyond UFO-MGen.

In practice, the intended workflow is:

```text
generated CIFs
    -> CHGNet / MACE / MatterSim
    -> relaxation
    -> formation-energy check
    -> phonon stability
    -> 300 K MD
    -> MSE decision for each backend
    -> selected if >= 2 backends classify the candidate as stable
```

```bash
python examples/mse_funnel.py
```

The shared MLIP interface is implemented in
[`ufo_mgen.mlip`](src/ufo_mgen/mlip/), with the stability gates and selection
logic in [`ufo_mgen.evaluation.mse`](src/ufo_mgen/evaluation/mse.py).
The harness is intended to automate and standardize MLIP-based screening; it
does not remove the underlying physical calculations, and DFT can still be
used as an independent higher-accuracy validation step for selected
candidates. Numerical manuscript outcome tables and large per-structure
evaluation outputs are not duplicated in this repository.

## Representative DFT-validated candidates

Six representative UFO-MGen-generated crystals shown in the manuscript are
included under
[`examples/dft_validated_candidates/`](examples/dft_validated_candidates/):

`AlPS4`, `KLiSe`, `Nb3Si`, `Pd4Se`, `ScGa3`, and `YGa3`.

These structures were independently checked with first-principles DFT
structural relaxation, formation-energy calculations, DFT phonons, and 300 K
NVT ab initio molecular dynamics (AIMD). The validation results are reported in
main-text Fig. 2(f) and Supplementary Sections S4.1-S4.3.

The repository provides the representative generated CIFs. Raw DFT outputs,
phonon working directories, and AIMD trajectories are not redistributed here.

## MP20 benchmark and extrapolation

The MP20 release is described by
[`configs/mp20.yaml`](configs/mp20.yaml) and
[`manifests/MP20_models.csv`](manifests/MP20_models.csv). The manifest
contains 3 shared union checkpoints and 161 per-space-group checkpoints.

```bash
python examples/mp20_benchmark.py
```

Evaluation utilities are available under
[`ufo_mgen.evaluation`](src/ufo_mgen/evaluation/). The extrapolation
implementation is in
[`ufo_mgen.generation.generate_extrapolation`](src/ufo_mgen/generation/generate_extrapolation.py).

## UFO-Mech

[`ufo_mgen.mech`](src/ufo_mgen/mech/) contains the property-guided extension
used for mechanical-property-oriented generation. The fine-tuning score uses
bulk, shear, and Young's moduli with equal weight:

```text
score = z(log K) + z(log G) + z(log E) - penalty
```

The lightweight surrogate is used only for candidate prioritization. Final
`K`, `G`, and `E` values are recomputed with MatterSim.

```bash
python examples/mech_finetune.py
```

## Pretrained weights

UFO-MGen model weights are not stored in GitHub.

| Archive | Contents |
|---|---|
| `UFO_MGen_Final_weights.tar.zst` | Stage I x1, Stage II x1, Stage III x120 |
| `UFO_MGen_MP20_paper_weights.tar.zst` | 3 shared union + 161 per-space-group checkpoints |

Checkpoint filenames, sizes, SHA256 hashes, and archive mappings are recorded
in:

- `manifests/UFO_MGen_models.csv`
- `manifests/MP20_models.csv`

The manifest DOI fields remain `PENDING` until the external archives are
deposited.

## Repository layout

```text
configs/       experiment and benchmark YAML configurations
data/          processed UFO-MGen dataset and route inventory
examples/      minimal usage examples and representative CIFs
manifests/     checkpoint inventories and SHA256 hashes
src/ufo_mgen/
  models/      HTS, COM, and SWG
  wyckoff/     encoding, decoding, canonicalization, lattice projection
  generation/  route-based and extrapolation generation utilities
  screening/   post-generation screening
  evaluation/  validity, coverage, S.U.N., MSE, interpolation/extrapolation
  mlip/        CHGNet, MACE, and MatterSim interfaces
  mech/        UFO-Mech utilities
  train/       training utilities and entry points
tests/         lightweight integrity and interface checks
```

## Tests

The repository includes lightweight tests that do not require the UFO-MGen
checkpoint archives:

```bash
python tests/test_wyckoff.py
python tests/test_decode.py
python tests/test_dataset.py
python tests/test_models.py
python tests/test_screening.py
python tests/test_mech.py
python tests/test_extrapolation.py
python tests/test_evaluation.py
```

## Release scope

This repository intentionally keeps the public release compact. Large generated
corpora, manuscript result tables, raw DFT calculation directories, and
third-party pretrained potentials are not redistributed here.

## License

MIT - see [`LICENSE`](LICENSE).
