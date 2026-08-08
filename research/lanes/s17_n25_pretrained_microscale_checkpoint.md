# S17-N25 — PRETRAINED_MICROSCALE_CHECKPOINT

- **Verdict:** **FAIL_CLOSED — no executable or prerequisite candidate.**
- **Audit time:** 2026-08-09 00:37–00:47 KST (bounded intake; written before minute 15).
- **Question:** Is there a pinned, pre-2026-07-05, commercially eligible pretrained checkpoint with a fixed issued-NWP-plus-terrain contract, demonstrated unseen-terrain use, a literal 117 m wind output, and unchanged Darwin arm64 execution without analyst-selected domain or preprocessing?
- **Restrictions observed:** documentation/model-card/paper metadata and nonmutating local inventory only; no repository/code/weight/data body download, install, project/model import, training, inference, or competition value/result access.

## Conjunctive gate

Every clause is mandatory; absence of literal evidence fails the candidate.

| Candidate | Pinned eligible checkpoint | Exact issued-NWP + terrain contract | Unseen terrain | Literal 117 m output | Unchanged Darwin arm64, zero free choices | Disposition |
|---|---|---|---|---|---|---|
| FuXi-CFD v1.0 | **No:** pinned ONNX weights exist, but CC BY-NC 4.0 forbids the required commercial use | **No:** tensor shapes are fixed, but issuance/source, farm-centred crop/regrid, DEM/roughness preparation are not | Yes | **No literal contract** | **No** | reject |
| WindSeer v1.0.1 | **Unverified:** official release establishes code, not a packaged pretrained checkpoint | **No:** sparse contemporaneous wind observations + terrain distance field, not issued NWP + terrain | Yes | **No literal contract** | **No** | reject |
| TerraWind | **No anonymous pinned asset:** pinned v2 is restricted; the open page is mutable `latest` v5 | **No:** station information, geography/density, terrain/wind inputs; no fixed issued-NWP schema | Not literally established for the target contract | **No:** near-surface product only | **No** | reject |
| CorrDiff Taiwan v1 | A pretrained package exists; commercial checkpoint terms were not established here | **No:** fixed Taiwan ERA5/WRF regional mapping; GFS is only an unevaluated example, not a terrain-conditioned Korea contract | **No** | **No:** approximately 3 km fields including 10 m wind | **No** | reject |

## Primary evidence

### FuXi-CFD — closest functional match, licence/runtime/output fail

- [directly_supported] The versioned Zenodo record was published 2026-02-25 as **v1** and identifies the official pretrained model; it states that **all data are CC BY-NC 4.0**. The pinned model card likewise says **CC BY-NC 4.0**. This directly contradicts commercial eligibility. [1–3]
- [directly_supported] The pinned card fixes inputs as `dem` and `roughness` (300×300) plus coarse `u_100m` and `v_100m` (9×9), and emits `u/v/w/k` on 27 non-uniform levels. [3]
- [directly_supported] The paper describes evaluation on unseen terrain and cross-region use, so unseen-terrain applicability alone is supported. [1]
- [unverified] Neither the pinned card nor paper specifies **117 m** as an output level. The card does not enumerate the 27 heights. Substituting/interpolating another height is forbidden analyst preprocessing. [1,3]
- [near_match_only] The card does not fix the issued forecast product/cycle, farm-centred 9 km crop, NWP-to-1 km regrid, DEM crop, or land-cover-to-roughness mapping. Those are free domain/preprocessing choices, not an executable contract. [1,3]
- [contradicts_premise] The checkpoint is ONNX, while unchanged local runtime contains no ONNX Runtime distribution. [3; local inventory]

### WindSeer — unseen terrain, but code-only and observation-conditioned

- [directly_supported] The paper says it operates on previously unseen topography without retraining. [4]
- [contradicts_premise] Its four inputs are a measurement mask, terrain distance field, and sparse horizontal **observed** winds; this is not an issued-NWP-plus-terrain contract. Output grid size/resolution follows the selected input grid rather than a fixed 117 m level. [4]
- [unverified] The paper's Code Availability points to source code, the tagged Zenodo record describes the software release, and the project README presents training/evaluation tools. None of these official pages identifies a pinned pretrained weight file. Train-yourself code receives no credit. [4–6]
- [contradicts_premise] Official setup targets Ubuntu with PyTorch (and CUDA when a GPU is present); documented embedded execution uses NVIDIA hardware. The unchanged local environment has neither PyTorch nor NVIDIA/CUDA. [4,5; local inventory]

### TerraWind — archived model near-match, but no admissible deployment contract

- [directly_supported] Zenodo record `12715088` is immutable v2, CC BY 4.0, but its files are restricted. The open v5 presentation is reached through mutable `/latest`; it lists one archive and checksum but does not supply an immutable v5 record URL on the inspected page. Anonymous pinned retrieval therefore is not established. [8,9]
- [contradicts_premise] The paper/project describe GraphNet use of discrete station information and AdaptNet use of geographic location/station density; the README demonstration uses random inputs and says different data can be handled by retraining. That is neither issued-NWP-only conditioning nor zero-choice deployment. [7,10]
- [contradicts_premise] The stated target is a near-surface wind field; no 117 m output is specified. [7,10]
- [contradicts_premise] The project requires installing its requirements and a deep-learning runtime; those are absent locally and dependency changes are forbidden. [10; local inventory]

### CorrDiff Taiwan — pinned checkpoint, wrong scale/domain/output

- [directly_supported] NVIDIA NGC lists **Modulus Checkpoints: CorrDiff**, Version 1, as pretrained checkpoints. [11]
- [contradicts_premise] The versioned Earth2Studio example says the Taiwan model maps quarter-degree global data to approximately 3 km, was trained on ERA5 and WRF over Taiwan, and has not evaluated the demonstrated GFS/out-of-training-year application. It outputs regional fields including `u10m/v10m`, not 117 m microscale wind. [12]
- [contradicts_premise] This is a Taiwan-fixed learned regional mapping, not demonstrated unseen-terrain deployment in Korea and not an explicit terrain input contract. Its workflow depends on Earth2Studio and PyTorch; neither is installed locally. [12; local inventory]

## Nonmutating local runtime inventory

Commands used only OS/hardware and installed-distribution metadata: `uname -srm`, `sw_vers`, `sysctl`, `system_profiler SPDisplaysDataType`, `.venv/bin/python --version`, `file .venv/bin/python`, and `uv pip list --python .venv/bin/python` (no model/package import).

- [directly_supported] OS: `Darwin 25.5.0 arm64`; macOS 26.5 (25F71).
- [directly_supported] Hardware: Apple M1, 16 GiB RAM, integrated 7-core Apple GPU/Metal 4; no NVIDIA device or CUDA runtime identified.
- [directly_supported] Project interpreter: Python 3.12.0, Mach-O arm64.
- [directly_supported] Relevant installed distributions found: NumPy, pandas, SciPy, scikit-learn. No distribution matching PyTorch, ONNX/ONNX Runtime, NVIDIA/CUDA, PhysicsNeMo/Modulus, Earth2Studio, GDAL, NetCDF4, PyProj, or h5py was present.

## Decision

[derived] No candidate satisfies the mandatory clauses jointly. FuXi-CFD is the only pinned model that directly supports both microscale 3-D output and unseen terrain, but its noncommercial licence independently closes it; it also lacks a literal 117 m/output-and-preprocessing contract and the unchanged runtime. WindSeer lacks an evidenced pretrained checkpoint and requires observations. TerraWind lacks anonymous pinned retrieval and an issued-NWP/117 m contract. CorrDiff is kilometre-scale and region-fixed. **Candidate count: 0; no handoff for execution or prerequisite work.**

## Sources (12 primary official project/model-card/paper URLs)

1. Nature Communications, FuXi-CFD paper (published 2026-03-09): https://www.nature.com/articles/s41467-026-70562-5
2. Zenodo, FuXi-CFD Dataset and Pre-trained Model v1: https://zenodo.org/records/18770845
3. Hugging Face, pinned FuXi-CFD model card (`6e44484…`): https://huggingface.co/linchensen/FuXi-CFD-model/blob/6e44484ff2c11efc1dea3a227942fe596e5c8931/fuxicfd-model/README.md
4. Nature Communications, WindSeer paper: https://www.nature.com/articles/s41467-024-47778-4
5. ETH Zürich WindSeer project, tag v1.0.1: https://github.com/ethz-asl/WindSeer/tree/v1.0.1
6. Zenodo, WindSeer v1.0.1: https://zenodo.org/records/10844690
7. Geophysical Research Letters, TerraWind paper: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024GL112124
8. Zenodo, pinned TerraWind v2 record: https://zenodo.org/records/12715088
9. Zenodo, TerraWind open latest presentation (v5 at audit): https://zenodo.org/records/12715088/latest
10. TerraWind author project: https://github.com/LLXxyI/TerraWind
11. NVIDIA NGC, CorrDiff inference package v1: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/corrdiff_inference_package
12. NVIDIA Earth2Studio 0.3.0, CorrDiff Taiwan inference documentation: https://nvidia.github.io/earth2studio/v/0.3.0/examples/04_corrdiff_inference.html
