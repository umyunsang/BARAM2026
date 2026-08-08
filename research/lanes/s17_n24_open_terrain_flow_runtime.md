# S17-N24 — open terrain-flow runtime intake (root fail-closed reconstruction)

## Verdict

- [derived] **`NO_READY_OPEN_TERRAIN_FLOW_RUNTIME`; selected candidates: `0`.** Several pre-cutoff open or commercially usable terrain-wind engines exist, but none simultaneously supplies a coefficient-complete zero-variant BARAM treatment, an unchanged Apple-M1 runtime, and a directly supported turbine-to-KPX-group output mapping.
- [directly_supported] The delegated reader completed its bounded documentation/runtime audit before its designated write was durable. The root therefore used no delegated conclusion as authority and independently reproduced the official repository/documentation and local-runtime facts below.
- [contradicts_premise] Open licensing alone does not make a transform executable. WindNinja/WindMapper expose substantive resolution, direction-category, initialization, vegetation, height, diurnal and averaging choices; GWOCSS requires a new Fortran toolchain and domain configuration; the Helbig repository labels itself unofficial/ongoing and exposes length/window choices; WindSeer documents training/evaluation tools and an Ubuntu/PyTorch/GDAL stack rather than a pinned zero-choice checkpoint contract.
- [derived] No code/data/raster was downloaded or run, no dependency changed, and no acquisition/install prerequisite is emitted. Installation would not cure the independent formula/mapping degrees of freedom.

## Scope

- [directly_supported] Root documentation checks used six primary official/project URLs, below the lane cap of ten, on 2026-08-09 KST. No repository clone, package, binary, model weight, DEM/raster body, data/API endpoint, inference, label, `actual_kwh`, 2024/test value, fit, prediction, policy, metric, score, Dacon or account action was used.
- [directly_supported] Local inspection was limited to executable/module presence. The project host is `Darwin arm64`; `WindNinja_cli`, `gdal-config`, `gfortran`, `mpirun`, and `cdo` are absent. In the unchanged project interpreter, `osgeo`, `pyproj`, `netCDF4`, `h5py`, `torch`, `rasterio`, and `mpi4py` are absent.
- [directly_supported] The frozen N20 DEM asset was not opened. N23's site-local/group-reducer, roughness, and wake blockers remain controlling inputs; this lane did not retune `terrain__sx300_h8_mean16`.

## Candidate matrix

| Candidate | Direct official/project facts | Zero-choice/runtime audit | Decision |
|---|---|---|---|
| **WindNinja** | [directly_supported] The official repository describes a diagnostic wind model for wildland-fire use. Its licence file places federal work in the public domain and external contributions under BSD terms. The README lists Boost, NetCDF, GDAL/PROJ/GEOS/CURL, Qt and OpenFOAM dependencies. | [contradicts_premise] The executable and core geospatial dependencies are absent. The related WindMapper documentation exposes selectable initialization, input/output height, mesh resolution, vegetation and diurnal settings rather than one provider-prescribed BARAM configuration. | **REJECT** |
| **WindMapper** | [directly_supported] Official docs state that it precomputes WindNinja wind fields and requires a projected, no-missing DEM plus a `WindNinja_cli` path. Configuration includes `res_wind`, `ncat`, a WindNinja config, `wind_average` and `targ_res`. | [contradicts_premise] Choosing direction/speed categories, 150 m versus another resolution, `mean_tile` versus `grid`, target averaging scale, vegetation, input height and domain is a multi-variant model family; GDAL/WindNinja/MPI are absent. | **REJECT** |
| **GWOCSS** | [directly_supported] The GPL-3 project describes a diagnostic complex-terrain model that can downscale prognostic model output. It requires autotools plus a Fortran/GCC toolchain and domain-specific runtime/localization, topography and meteorological input files. | [contradicts_premise] `gfortran` is absent and the project itself requires production-specific input preparation/configuration. Neither a 117 m turbine treatment nor one KPX-group reducer is fixed. | **REJECT** |
| **Helbig implementation** | [directly_supported] The MIT repository calls itself an **unofficial**, **ongoing** implementation of Helbig et al. and its example requires `dx`, length `l`, and explicit x/y window sizes. | [contradicts_premise] Those are material free choices, the implementation is not a primary reproduction oracle, and its static factor does not cure the N23 target-site/group-reduction contract. | **REJECT** |
| **WindSeer** | [directly_supported] The BSD-3 repository (created 2023) provides neural-network training/evaluation tools for volumetric wind prediction. Its README specifies Ubuntu, PyTorch (CUDA optional) and a long GDAL/NetCDF/HDF5/pyproj stack. | [unverified] The README does not identify a pinned commercially eligible pretrained checkpoint plus fixed BARAM input/domain/output contract. `torch`, GDAL, NetCDF/HDF5 and pyproj are absent locally. | **REJECT** |

## Exact-treatment gate

A qualifying treatment would have to fix, from primary evidence rather than analyst choice: (1) the model and version, (2) domain/buffer/projection/grid resolution, (3) vegetation/roughness and stability mode, (4) coarse issued-wind initialization and height, (5) boundary conditions and solver/averaging settings, (6) 117 m extraction at all 17 turbines, and (7) one source-prescribed aggregation into KPX groups 1–3. [derived]

- [directly_supported] The reviewed WindMapper configuration alone leaves at least resolution, `ncat`, initialization configuration, input/output height, vegetation, diurnal mode, averaging method and target resolution selectable.
- [directly_supported] GWOCSS and WindSeer require absent compiled/ML/geospatial stacks; WindSeer additionally documents training tools rather than a pinned inference asset in the reviewed README.
- [directly_supported] The Helbig example literal `downscale(alti, dx=30, l=2000, x_win=69//2, y_win=79//2)` demonstrates multiple domain/window parameters, not a coefficient-free transform.
- [derived] A dependency prerequisite cannot be selected because successful installation would still leave these physical/domain/group choices unresolved. A generic terrain output would also be a variant of the already closed terrain family unless every N23 site-mapping blocker were independently cured.

**Final selection: `NONE`.** [derived]

## Primary official/project URL ledger

1. [directly_supported] WindNinja official repository/README: https://github.com/firelab/windninja
2. [directly_supported] WindNinja official licence: https://github.com/firelab/windninja/blob/master/LICENSE
3. [directly_supported] WindMapper official configuration documentation: https://windmapper.readthedocs.io/configuration.html
4. [directly_supported] GWOCSS official project repository: https://github.com/sarnold/gwocss
5. [directly_supported] `helbigwindparam` project repository: https://github.com/louisletoumelin/helbigwindparam
6. [directly_supported] WindSeer official repository/README: https://github.com/ethz-asl/WindSeer

## Reproducibility hashes of fetched documentation text

```json
{
  "windninja_repo": "0fb3dacdfb2f8f5116a95c2489923df720e6101e3c4b05b152fc6c209325f3dc",
  "windninja_license": "27f612e51370b135e34cbd1750a233ddc6d8e7a55faf292efb8ca034f1457957",
  "windmapper_config": "d7e48a6560abce33abb0f57088dd4ed63b088e63805161d4c9ea8343d68d1e27",
  "gwocss_repo": "586b3acfba763ff0422e1599cf3a4f62cc4b3db31980d00e7f40c04949e7b78c",
  "helbig_repo": "dd3c43bd72821245967c4aa6983205c636371141b9f99c38d7a0a76242568e19",
  "windseer_repo": "86f3d2083529b43392767d7e7722bcb23ee19b59785da4d0768d04b3d3444311"
}
```
