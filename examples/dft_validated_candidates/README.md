# Representative DFT-validated candidates

This directory contains the six UFO-MGen-generated crystal structures shown as representative examples in the main manuscript:

- AlPS4
- KLiSe
- Nb3Si
- Pd4Se
- ScGa3
- YGa3

These candidates were independently validated with first-principles calculations following the manuscript/Supplementary Information protocol: full DFT structural relaxation, formation-energy evaluation, DFT phonons, and 300 K NVT ab initio molecular dynamics (AIMD). The corresponding validation results are reported in main-text Fig. 2(f) and Supplementary Sections S4.1-S4.3.

The CIFs here are the representative UFO-MGen-generated structures supplied for inspection and reuse; the DFT validation outputs and trajectories are reported in the manuscript/SI rather than duplicated in this repository.

Note: these CIFs are explicit-coordinate pymatgen exports and therefore carry a `P 1` CIF header with only the identity symmetry operation. This serialization header should not be interpreted as the crystallographic classification of the candidate; symmetry should be re-detected from the coordinates using the tolerances described in the workflow.
