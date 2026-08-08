# S17-N18 — local pretrained-NWP executability audit

## 0. Lane contract and verdict

- This is the bounded `pretrained_nwp` lane for `S17-N18_POST_RWA_FRONTIER_RESEARCH_INTAKE`; the only repository write made by this lane is this report. [directly_supported]
- Official metadata retrieval and the final nonmutating host check were completed by `2026-08-08T13:33:19Z`. [directly_supported]
- Retrieval/audit scope was nine official source packages plus eleven named local documents, within the requested ceiling of ten official packages. [directly_supported]
- The lane fetched only documentation, repository/API metadata, HTTP `HEAD` responses, and NOAA GRIB `.idx` text; it downloaded no checkpoint, model binary, GRIB body, NetCDF body, or other data body. [directly_supported]
- The lane ran no model, inference, fit, scorer, test, install, dependency resolution, browser/account mutation, or remote API inference, and accessed no underlying test/2024/lockbox feature, label, prediction, model, score, rejected, or quarantined body. [directly_supported]

**Verdict: `NO_EXECUTABLE_LOCAL_PRETRAINED_NWP_IN_UNCHANGED_ENVIRONMENT`.** [derived]

- FourCastNet v1, Aurora, and AIFS Single v1 each have a pre-`2026-07-05` weight object with a licence that permits commercial use, but none is runnable in the unchanged project environment. [derived]
- FourCastNet v1 is the only candidate audited here that combines an exact official GFS field adapter with native 100 m wind output and the smallest development-state payload, but its official runtime is Linux/NVIDIA-GPU oriented, its Earth2Studio model badge says `gpu:40gb`, and this host is a 16 GiB Apple-M1 macOS machine with none of its runtime packages installed. [contradicts_premise]
- Aurora has a structurally complete GFS field inventory, but the applicable pretrained checkpoint is 5.026 GB, the official local example moves it to CUDA, the 451 MB small checkpoint is expressly “only ... for debugging,” and neither checkpoint/runtime is present locally. [contradicts_premise]
- AIFS Single v1 has native 100 m output and a 994 MB checkpoint, but its official recipe requires ECMWF-specific initial fields/regridding and CUDA/FlashAttention; the recipe says CPU use is “tricky,” and GFS lacks an official exact adapter for its surface/static/soil contract. [contradicts_premise]
- Therefore legal eligibility in the named prior report does not imply local executability; no weight acquisition or inference prerequisite should be promoted from this lane. [derived]

## 1. Evidence tags and decision gates

- `[directly_supported]` denotes an exact official quote, provider metadata response, or nonmutating local-machine observation. [directly_supported]
- `[derived]` denotes arithmetic or a decision obtained only from directly supported premises. [directly_supported]
- `[near_match_only]` denotes a structural/name match whose exact source semantics or supported use differ. [directly_supported]
- `[contradicts_premise]` denotes evidence that defeats the premise that an artifact is executable here. [directly_supported]
- `[unverified]` denotes a gap that this bounded no-download audit did not close. [directly_supported]
- A candidate must pass all six gates: pre-cutoff **weight object**, commercial licence, basis-safe fixed issue/lead, fixed development coverage and fields, local runtime, and task-relevant output. [derived]
- A model announcement or software release date is not a checkpoint publication date; the dates are reported separately below. [derived]

## 2. Unchanged-host runtime gate — the smallest prerequisite audit

The smallest prerequisite audit was the zero-download host/package manifest below, and it already fails before any weight acquisition. [derived]

| Fact observed nonmutatingly | Result | Tag |
|---|---|---|
| OS / architecture / CPU | macOS 26.5, `arm64`, Apple M1 | [directly_supported] |
| Memory / GPU | 17,179,869,184 bytes (16 GiB); integrated Apple M1 7-core Metal GPU; no NVIDIA CUDA device | [directly_supported] |
| Free project-volume space | about 45 GiB at audit time | [directly_supported] |
| Project interpreter | `.venv/bin/python`, Python 3.12.0 | [directly_supported] |
| Absent modules | `torch`, `modulus`, `earth2studio`, `aurora`, `onnxruntime`, `onnx`, `netCDF4`, `h5py` | [directly_supported] |
| Present relevant modules | `cfgrib 0.9.15.1`, `eccodes 2.47.0`, `xarray 2026.7.0`, `pydantic 2.13.4` | [directly_supported] |
| Core project declaration | `pyproject.toml` has no PyTorch/PhysicsNeMo/Earth2Studio/Aurora/Anemoi runtime dependency | [directly_supported] |

- Canonical JSON over the host observations has SHA-256 `0eb6d0fca50197a8fc38f04118e0ccbf50c2a5ab4be70fbde3343e36f2a3b4a5`. [directly_supported]
- Merely importing any of the three candidates would require a dependency mutation, which this task forbids and the “unchanged project environment” premise excludes. [derived]
- Storage capacity alone is not an execution PASS because accelerator type, working memory, and installed runtime are independently binding. [derived]

## 3. Basis-safe GFS state, lead, field, and development-coverage audit

- The named local eligibility report fixes the cutoff at `D-1 05:00 UTC` and the required valid-time interval at `D-1 16:00 UTC ... D 15:00 UTC`. [directly_supported]
- A fixed `D-1 00Z` GFS run can provide forecast states `f006` and `f012`, both generated from the same already-issued run rather than observations at their future valid times. [derived]
- With `f012` as the current state, a six-hour model needs outputs at model steps `+6,+12,+18,+24,+30 h`; interpolation between the `f012` state and those outputs covers the required hourly valid interval. [derived]
- Aurora and AIFS require the two history states `f006,f012`; FourCastNet requires one state and can use `f012`. [directly_supported]

### 3.1 Deterministic NOAA metadata receipt

- URL family audited was `https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.YYYYMMDD/00/atmos/gfs.t00z.pgrb2.0p25.f{006,012}[.idx]` for every date from `2023-04-01` through `2023-12-31`. [directly_supported]
- All `275 × 2 = 550` GRIB objects returned HTTP 200 to `HEAD`; every GRIB `Last-Modified` time was on its issue date and no later than `03:50:33 UTC`, more than one hour before the `05:00 UTC` basis cutoff. [directly_supported]
- Canonical JSON over the 550 GRIB `status/content-length/Last-Modified/ETag` tuples has SHA-256 `46972ba3a9af020ecb3f63f9b70580bf4748f2c6bf9d18dceb5bf9303375c42b`. [directly_supported]
- All 550 `.idx` objects returned HTTP 200, and their 22,464,172 bytes of index text each contained 743 records. [directly_supported]
- Canonical JSON over each index URL/status/body-SHA tuple has SHA-256 `3ee1c17d549af368f4f35bf0d627a09c884ff76d92fcbc463a2ab6c9f1fe228d`. [directly_supported]
- The `2023-05-01 f012.idx` object was rebuilt on `2023-05-17`, but its referenced GRIB body itself has `Last-Modified: Mon, 01 May 2023 03:33:56 GMT`; chronology is therefore taken from the immutable forecast object, not the later index rewrite. [directly_supported]
- Three representative index bodies and hashes were `20230401/f006` `389cefb28e8e5d56ed652e2c705c094715737304e7ae143225ea594402565bd9`, `20230701/f012` `32138470ae5250673ba8b800c3c3ab726ac33975346a5c39c6e47741acdeba58`, and `20231231/f012` `71441d56e164a234071984cb2b094948eb810a5067ec249dec3433d6f34dd2aa`. [directly_supported]

### 3.2 Exact inventory and byte-range payload

| Candidate contract | Daily states | Inventory result on all 275 development dates | Exact selected byte-range payload for 2023-04-01...12-31 | Output relevance | Tag |
|---|---:|---|---:|---|---|
| FourCastNet v1: 26 fields | `f012` | 26/26 exact records on every state | 6,052,641,126 bytes (median 22,158,016 bytes/day) | native `u100m,v100m`, six-hour steps | [directly_supported] |
| Aurora pretrained: 4 surface + 5 variables × 13 pressure levels = 69 fields/state | `f006,f012` | 69/69 exact records on every state | 32,972,909,461 bytes (two states/day) | `10u,10v`, but no documented native 100 m output | [directly_supported] |
| AIFS Single v1: 12 named surface/static + four soil-layer fields + 6 variables × 13 pressure levels | two six-hour histories | no official GFS adapter; exact `tcw/slor/sdor` and soil semantics were not established | deliberately not estimated | native `100u,100v` output | [near_match_only] |

- The official Earth2Studio GFS lexicon maps all 26 FourCastNet names to exact GFS records and multiplies GFS `HGT` by `9.81` for model geopotential. [directly_supported]
- The same conversion is needed for Aurora’s `z`; the official Aurora recipe uses geopotential, while the NOAA index exposes geopotential height. [derived]
- Field existence and basis timing make forecast-state substitution technically constructible for FourCastNet and Aurora, but neither official model card validates skill after replacing its training/operational initial-condition distribution with GFS `f012`; that applicability remains only a near match. [near_match_only]
- No production/test-period object was queried to make these coverage claims; production symmetry would have to reuse the frozen `00Z/f006/f012`, variable dictionary, units, grid, and interpolation recipe without inspecting outcomes. [derived]

## 4. Candidate audit

### 4.1 NVIDIA FourCastNet v1

#### Date, object, licence, and size

| Distinct fact | Evidence | Tag |
|---|---|---|
| Model release date | NVIDIA card: `Release Date: October 25, 2023` | [directly_supported] |
| Public HF repository/object date | repository `createdAt=2026-03-02T16:31:44Z`; `fcn.mdlus` last commit `c67a639...` at `2026-03-02T16:57:32Z` | [directly_supported] |
| Weight identity | 301,168,640 bytes; LFS SHA-256 `995cdfdc3b64330caade5518aff09e0ce8f941b4262f3f8eb792b6fea8b6423a` | [directly_supported] |
| Pinned repository revision | `nvidia/fourcastnet1@f63c56bd37c3fad04836422fd5a3a10329f95141`, `2026-03-05` | [directly_supported] |
| Licence | card says “ready for commercial/non-commercial use” and links Apache 2.0 | [directly_supported] |
| Minimal package bytes | 301,176,583 bytes for checkpoint, two 336-byte statistics arrays, and metadata/card files | [derived] |

- The 2023 model release date does **not** by itself date the currently auditable checkpoint; the auditable weight object appeared in the HF history in March 2026, still before the July 2026 cutoff. [derived]
- The card lists the exact 26-variable 0.25° `720 × 1440` input and output contract and includes `u100m,v100m`. [directly_supported]
- The pinned Earth2Studio `0.16.0` release was published `2026-06-29`, before the cutoff, and its FCN loader pins the same March-2026 checkpoint object. [directly_supported]

#### Runtime closure

> “Supported Hardware Microarchitecture Compatibility: NVIDIA Ampere; NVIDIA Hopper; NVIDIA Turing” and “Supported Operating System: Linux.” [directly_supported]

- Earth2Studio `0.16.0` declares `torch>=2.5.0`, adds `nvidia-physicsnemo>=1.0.1` for FCN, classifies the environment as `GPU`, and labels FCN `gpu:40gb`. [directly_supported]
- The local host is macOS/Apple-M1 with 16 GiB unified memory and has none of `torch`, `earth2studio`, or PhysicsNeMo installed. [contradicts_premise]
- Consequently FourCastNet v1 is not locally runnable in the unchanged environment even though licence, cutoff, basis timing, field inventory, and native 100 m output pass. [derived]
- A historical GFS development acquisition would add 6.053 GB of selected ranges to the 0.301 GB package, but downloading either is nonexecutive because the runtime gate already failed. [derived]

### 4.2 Microsoft Aurora

#### Date, object, licence, and size

| Distinct fact | Evidence | Tag |
|---|---|---|
| First official code release receipt | GitHub release `v0.0.0`, `2024-07-25`; stable `v1.0.0`, `2024-08-21` | [directly_supported] |
| Runtime pin used by this audit | code tag `v1.7.0`, release `2025-06-23`, commit `7765cd44c63427e310285dd1768a7800cad262b6` | [directly_supported] |
| Weight-repository pin | `microsoft/aurora@d8753da77552eab0f27150c0efb91cb3acb425d5`, commit `2025-06-20` | [directly_supported] |
| Full pretrained weight object | `aurora-0.25-pretrained.ckpt`, 5,025,579,446 bytes, SHA-256 `64e6c62a7043498292a6462e8c3a08171300bc166cba6b2b557021c4b684a4c9`, committed `2024-08-21` | [directly_supported] |
| Fine-tuned weight object | 5,038,339,714 bytes, SHA-256 `25e429a3d615b4e2ef449fbb433eddaffbcc9c3678306585ef9567721bde6146`, committed `2024-08-21` | [directly_supported] |
| Small/debug weight object | 451,339,106 bytes, SHA-256 `f80f78de1524a9faba8c9053e4a8ce6a2114ec01cff7f7b4efe9377200d50621`, committed `2024-08-21` | [directly_supported] |
| Static file | `aurora-0.25-static.pickle`, 12,459,115 bytes, SHA-256 `e382103f6b24bcf1f996cc0af217c71ff2fc66507a5221e1300b5017581bd318` | [directly_supported] |
| Licence | HF card and code are MIT; the licence permits use, modification, distribution, sublicensing, and sale | [directly_supported] |

- The model/code release receipts and the individual checkpoint-object commits are separate facts; all applicable objects above predate the cutoff. [derived]
- The later Aurora-v1.5 weight files added to the repository after the cutoff are outside this audit and are not needed for the disposition. [directly_supported]

#### Field and runtime closure

- The official pretrained recipe uses two histories of `2t,10u,10v,msl`, static `lsm,z,slt`, and `z,u,v,t,q` at 13 pressure levels; the development GFS indices contain the 69 dynamic records at both `f006` and `f012`. [directly_supported]
- The official ERA5 notebook says, “The fine-tuned version of Aurora specifically only works with IFS HRES T0,” so the 5.038 GB fine-tuned checkpoint is not an official GFS-state path. [directly_supported]
- The applicable cross-source candidate is therefore the 5.026 GB pretrained checkpoint, not the fine-tuned checkpoint. [derived]
- The small class docstring says it “Should only be used for debugging,” so its 451 MB size cannot be used to claim an operational forecast candidate. [directly_supported]
- Aurora’s default surface outputs are the same `2t,10u,10v,msl` variables; native 100 m wind is not in the pinned default output contract. [directly_supported]
- The official local example executes `model = model.to("cuda")`; the package requires `torch`, `timm`, `einops`, `huggingface-hub`, `netcdf4`, and other dependencies that are absent here. [directly_supported]
- The full model class says it defaults to a 1.3-billion-parameter configuration, while this host has only 16 GiB unified memory and no CUDA device. [contradicts_premise]
- The nominal pretrained package plus the audited development input ranges is about 38.01 GB before decoding, regridding, working tensors, or outputs, leaving inadequate evidence of safe headroom on a volume with about 45 GiB free. [derived]
- Aurora is therefore nonexecutive here; even on another accelerator, GFS forecast-state substitution and lack of native 100 m output would remain task-specific applicability gaps. [derived]

### 4.3 ECMWF AIFS Single v1.0 — “another” commercially usable open-weight candidate

#### Date, object, licence, and size

| Distinct fact | Evidence | Tag |
|---|---|---|
| Model/package repository date | official HF repository `createdAt=2025-02-04`; card calls v1.0 the first operationally supported AIFS model | [directly_supported] |
| Pre-cutoff pin | `ecmwf/aifs-single-1.0@e417b59c21435f58b05139312d1e8d89fb01d8b6`, `2025-06-27` | [directly_supported] |
| Weight object | `aifs-single-mse-1.0.ckpt`, 994,084,883 bytes, LFS SHA-256 `1fed399c097c0127d5bbe074f4f8bbc123759736145d990699c215ff07543ccd`, committed `2025-02-20` | [directly_supported] |
| Licence | weights CC BY 4.0; notebook/scripts Apache 2.0 | [directly_supported] |
| Output relevance | model card lists 100 m horizontal wind components as outputs | [directly_supported] |

- Repository/model-version timing and checkpoint-object timing are again distinct, and both precede the cutoff. [derived]

#### Field and runtime closure

- The model takes atmospheric states at `t-6h,t0`, forecasts in six-hour increments, and requires 13-level `z,u,v,w,q,t` plus surface, soil, and static forcings. [directly_supported]
- Its official notebook retrieves ECMWF Open Data, converts a 0.25° lat/lon grid to `N320`, renames ECMWF soil fields, and converts geopotential height by `9.80665`; it supplies no GFS adapter. [directly_supported]
- GFS has many pressure-level near matches, but exact `tcw`, sub-grid-orography, and soil semantics were not established as identical, so field/coverage symmetry from GFS fails closed rather than being inferred from similar names. [near_match_only]
- The official install recipe adds `anemoi-inference[huggingface]`, `anemoi-models`, `earthkit-regrid`, `ecmwf-opendata`, and `flash_attn`; none is in the project environment. [directly_supported]
- The official runner uses `device="cuda"` and warns that CPU execution is “tricky” because FlashAttention supports NVIDIA/AMD GPUs, while the SDPA alternative uses much more memory. [directly_supported]
- On this Apple-M1 host, AIFS therefore fails both exact GFS-input and unchanged-runtime gates despite its attractive checkpoint size and native 100 m output. [derived]

## 5. Comparative disposition and smallest next audit

| Candidate | Cutoff | Commercial licence | Basis-safe fixed GFS states | Fixed dev fields | Unchanged local runtime | Native 100 m | Disposition | Tag |
|---|---|---|---|---|---|---|---|---|
| FourCastNet v1 | pass | pass, Apache 2.0 | pass, `00Z/f012` | pass, 26/26 × 275 | **fail**: Linux/NVIDIA/40 GB badge; packages absent | pass | close as nonexecutive | [derived] |
| Aurora pretrained | pass | pass, MIT | pass, `00Z/f006,f012` | structural pass, 69/69 × 275; source substitution near-match | **fail**: CUDA example; 1.3B/full runtime absent | fail | close as nonexecutive | [derived] |
| AIFS Single v1 | pass | pass, CC BY 4.0 | lead arithmetic could match | **fail**: no exact official GFS adapter | **fail**: CUDA/FlashAttention stack absent | pass | close as nonexecutive | [derived] |

- **Smallest metadata/runtime prerequisite audit:** `FCN-R0`, consisting only of the host manifest, import-spec check, pinned runtime manifest, checkpoint tree metadata, and GFS `.idx`/`HEAD` inventory, is the minimum because FourCastNet has the smallest exact task-relevant payload. [derived]
- `FCN-R0` has already been completed in this lane and failed at the local Linux/NVIDIA/40-GB-runtime gate before weights. [contradicts_premise]
- There is consequently **no next executable candidate** under the unchanged-environment contract; the correct bounded action is to acquire nothing and close the local-pretrained-NWP branch. [derived]
- If a future root-authorised scope supplies an already-existing Linux NVIDIA execution target with at least the model’s documented 40-GB class and a frozen pre-cutoff package lock, FourCastNet would be the first candidate to re-audit, but that is a changed environment and is not authority to install, download, or infer now. [derived]

## 6. Deterministic source and local-document ledger

**Official source-package count: 9.** [directly_supported]

| ID | Official package, pin, and exact URLs | Evidence acquired | Tag |
|---|---|---|---|
| S1 | NVIDIA FourCastNet HF package, `f63c56b...`: [card](https://huggingface.co/nvidia/fourcastnet1/raw/f63c56bd37c3fad04836422fd5a3a10329f95141/README.md), [tree API](https://huggingface.co/api/models/nvidia/fourcastnet1/tree/f63c56bd37c3fad04836422fd5a3a10329f95141?recursive=true&expand=true) | card SHA-256 `684a2f67...ee4bc5`; tree-response SHA-256 `cee945d1...78ccc2` | [directly_supported] |
| S2 | NVIDIA Earth2Studio `0.16.0` / `c99422c...`: [pyproject](https://raw.githubusercontent.com/NVIDIA/earth2studio/c99422c3e75ca29d200440ec1bd2d4ac8282683c/pyproject.toml), [FCN loader](https://raw.githubusercontent.com/NVIDIA/earth2studio/c99422c3e75ca29d200440ec1bd2d4ac8282683c/earth2studio/models/px/fcn.py), [GFS lexicon](https://raw.githubusercontent.com/NVIDIA/earth2studio/c99422c3e75ca29d200440ec1bd2d4ac8282683c/earth2studio/lexicon/gfs.py) | body SHA-256 values `8a04c3aa...8867b8`, `7d6205f5...cb8890d`, `979dfff8...026dc3` | [directly_supported] |
| S3 | Microsoft Aurora HF package, `d8753da...`: [card](https://huggingface.co/microsoft/aurora/raw/d8753da77552eab0f27150c0efb91cb3acb425d5/README.md), [tree API](https://huggingface.co/api/models/microsoft/aurora/tree/d8753da77552eab0f27150c0efb91cb3acb425d5?recursive=true&expand=true) | body SHA-256 `d9693825...ef8385`; response SHA-256 `70c3b2ca...013e6b27` | [directly_supported] |
| S4 | Microsoft Aurora code/docs `v1.7.0`: [README](https://raw.githubusercontent.com/microsoft/aurora/v1.7.0/README.md), [pyproject](https://raw.githubusercontent.com/microsoft/aurora/v1.7.0/pyproject.toml), [MIT licence](https://raw.githubusercontent.com/microsoft/aurora/v1.7.0/LICENSE.txt), [ERA5 example](https://raw.githubusercontent.com/microsoft/aurora/v1.7.0/docs/example_era5.ipynb) | body SHA-256 values `5ae5737e...8d292`, `bb640b8e...6574a5`, `131a2e30...d5adf`, `54141abc...e3c08` | [directly_supported] |
| S5 | NOAA GFS official [registry](https://registry.opendata.aws/noaa-gfs-bdp-pds/) and `noaa-gfs-bdp-pds` object/`.idx` endpoint family | complete 2023-04-01...12-31 `f006/f012` HEAD/index receipts; canonical hashes in §3 | [directly_supported] |
| S6 | ECMWF AIFS Single v1 HF package, `e417b59...`: [card](https://huggingface.co/ecmwf/aifs-single-1.0/raw/e417b59c21435f58b05139312d1e8d89fb01d8b6/README.md), [tree API](https://huggingface.co/api/models/ecmwf/aifs-single-1.0/tree/e417b59c21435f58b05139312d1e8d89fb01d8b6?recursive=true&expand=true), [inference notebook](https://huggingface.co/ecmwf/aifs-single-1.0/raw/e417b59c21435f58b05139312d1e8d89fb01d8b6/run_AIFS_v1.ipynb) | body/response SHA-256 `2d6125ca...f83b09`, `d67ba613...a30b4`, `8123e126...38b4e` | [directly_supported] |
| S7 | NVIDIA Earth2MIP `v0.1.0` / `e43be507...`: [pyproject](https://raw.githubusercontent.com/NVIDIA/earth2mip/e43be5074d920adc6bbc9a6fd3ff86bb05dd5178/pyproject.toml), [FCN adapter](https://raw.githubusercontent.com/NVIDIA/earth2mip/e43be5074d920adc6bbc9a6fd3ff86bb05dd5178/earth2mip/networks/fcn.py) | screened as the older checkpoint entrypoint; body SHA-256 `ca070260...1023fe`, `534ee2ae...33047` | [directly_supported] |
| S8 | [NVIDIA Modulus official GitHub API metadata](https://api.github.com/repos/NVIDIA/modulus) | runtime-family identity screen only; response SHA-256 `1c9bccdd...762820` | [directly_supported] |
| S9 | [Google DeepMind GraphCast official GitHub API metadata](https://api.github.com/repos/google-deepmind/graphcast) | alternative-family screen only; no eligibility claim relies on it; response SHA-256 `79f307a1...461c63` | [directly_supported] |

**Named local documents: 11.** [directly_supported]

| ID | Local document and SHA-256 | Use | Tag |
|---|---|---|---|
| L1 | `AGENTS.md`, `91a11f95f94e8de86f5c84b204e1893830b3ed5693e3d7ede7f2e351e9891e9c` | competition boundary and lane restrictions | [directly_supported] |
| L2 | `reports/s17_n18_post_rwa_frontier_intake_predeclaration.json`, `2f00184f84f990d06125f5b89cb174a6b1f15a3afd1094d51568203deb33312f` | question, source/time bound, forbidden actions, selection rule | [directly_supported] |
| L3 | `research/rwa_external_eligibility.md`, `958ea7a0b96371b6d773920038ec8d6e07e7903b4b26f5dd8443dd6d354bf4e5` | prior legal/time candidate list and target/cutoff arithmetic, re-audited here for runtime executability | [directly_supported] |
| L4 | `research/lanes/S15_sota_nwp.md`, `ab8328f5c05b46a32ba8f3259baf1ea9f2914659eceecb3c1d4b68b85c03db3e` | context screen only; no quantitative claim imported | [directly_supported] |
| L5 | `research/lanes/S12_ext_nwp_sources.md`, `c49a6b2a5165e8f6676f2b92b270eb192cd2047b2439dad0ebbd57e9474aeaa6` | context screen only; no restricted-period result imported | [directly_supported] |
| L6 | `reports/m271_cycle47_external_nwp_revived.md`, `bf2c0e204a51912ff8c2c062675d06237f4b7ca66ebac3300280ba0cec6950f7` | context screen only; no result imported | [directly_supported] |
| L7 | `reports/m271_cycle47_external_nwp_revived_receipt.json`, `77e95eaf10d9a0b203b2b19184cc2ae4aa4510eaf1e07d49028eaf4e4d3ca39b` | provenance screen only; no result imported | [directly_supported] |
| L8 | `reports/m271_cycle51_external_source_probe.md`, `03644657e058f77498008a1a7090b919ff37058d7719d5a8f0d30ada1e01f3ce` | context screen only; no result imported | [directly_supported] |
| L9 | `reports/m271_cycle51_external_source_probe_receipt.json`, `6b62a89ca5c45e7209c2859e3a5249da8574c3b963a31ab8ccb215c4bcb4116a4` | provenance screen only; no result imported | [directly_supported] |
| L10 | `reports/m271_cycle52_external_closure.md`, `4a5050e325372412876c5d111ebb7ccc84473292551f808ee321dcad2989cd67` | context screen only; no result imported | [directly_supported] |
| L11 | `reports/m271_cycle52_external_closure_receipt.json`, `1180be58bf21e5e72fa7a2326577188519bbd19f8388794af6d55930ca78db10` | provenance screen only; no result imported | [directly_supported] |

## 7. Final answer to the intake question

- No pre-`2026-07-05`, commercially usable open pretrained forecast model audited here is actually runnable locally from basis-safe historical GFS forecast states in the **unchanged** BARAM2026 environment. [derived]
- FourCastNet’s data contract is executable in principle on a different compliant Linux/NVIDIA host, Aurora’s GFS substitution is only a pretrained-domain near match, and AIFS lacks an exact official GFS adapter; none clears the present runtime gate. [derived]
- Source count is `9 official packages + 11 named local documents`; disposition is `CLOSE_NO_ACQUISITION`, with zero model/data/weight download and zero inference. [directly_supported]
