# S12 — External public NWP sources as a third forecast axis (2023 + 2024 archive)

**Lane:** read-only research (`research/` writes only, no fits, no lockbox, no repo mutation)
**Date:** 2026-08-07 (KST) · **Author:** S12 external-NWP research lane
**Question:** can any *external public* NWP product be added, legally and practically, as a third
forecast source alongside the competition-supplied LDAPS + GFS pair, with an archive covering
**both CY2023 (development) and CY2024 (test)** over ~34–38 N / 126–130 E?

---

## 0. Evidence-grade convention

| Grade | Meaning |
|---|---|
| **A — primary, machine-verified** | I issued the HTTP/S3 request myself in this session and quote the literal response (bucket listings, GRIB `.idx` inventories, `.index` JSON, `Last-Modified` headers, zarr metadata). Reproducible by re-issuing the same URL. |
| **B — primary, full-text read** | I fetched the full page and quote its own words (licence pages, documentation pages). |
| **C — snippet only** | Only a search-result snippet was seen; the full page was **not** read. Treat as a lead, not a fact. |
| **D — inference** | My arithmetic or reasoning on top of A/B facts. Marked inline. |

Anything **not** verified from full text is explicitly flagged `⚠ NOT VERIFIED`.

---

## 1. Bottom line

**Yes — but only two products actually clear all four gates (2023+2024 archive · commercial licence ·
not reanalysis · issued before the D-1 14:00 KST basis time), and the better of the two is NOAA GEFS.**

| | GEFS (NOAA) | ECMWF open data IFS |
|---|---|---|
| archive covers 2023 | ✅ from 2017-01-01 | ⚠ from **2023-01-18**, with a 6-day hole |
| archive covers 2024 | ✅ | ✅ |
| hub-height (100 m) wind | ✅ 100 m + 80 m u/v (0.5°) | ❌ in 2023 (0.4° set has no 100 m); ✅ from 2024-02-01 (0.25°) → **feature definition is not constant across dev and test years** |
| run available by D-1 05:00 UTC | ✅ 00Z of D-1 lands **03:51–04:04 UTC** | ❌ 00Z lands 07:34 UTC → must fall back to the **D-2 12Z** run (leads 27–50 h) |
| licence | NOAA NODD open, commercial OK | CC-BY-4.0, commercial OK (explicit) |
| ensemble spread available | ✅ 31 members | ✅ 51 members but ~376 GB to extract |

**Top recommendation: NOAA GEFS 100 m winds, 31 members, from the D-1 00Z cycle** — accessed either
by GRIB byte-range from `s3://noaa-gefs-pds/.../pgrb2bp5/` or (much cheaper) from the
dynamical.org Icechunk-Zarr mirror. **Its value is not a better conditional mean — it is the
ensemble spread**, i.e. a per-hour, per-site conditional-uncertainty signal, which is exactly the
quantity the settlement-action policy (`T*_G*`) needs and which neither LDAPS nor GFS supplies.
Main risk: GEFS shares the GDAS analysis and the FV3 core with GFS, so its *mean* is near-collinear
with the GFS feature block already in the model (the repo has already measured LDAPS↔GFS error
correlation ≈ 0.78 and found averaging gain 4.6 % vs ~11.0 % required — GEFS's mean will be worse,
not better, than that). **Any claim of gain must be tested on the spread channel, not the mean.**

---

## 2. The binding time constraint (derivation — grade D on A-grade inputs)

Official basis time **D-1 14:00 KST = D-1 05:00 UTC**.
Target hours: day D 00:00–23:00 KST = **D-1 15:00 UTC → D 14:00 UTC**.

Therefore a run initialised at time `R` is admissible iff `R ≤ D-1 05:00 UTC`, and the needed lead
times are `valid − R`:

| Candidate run | Measured availability (S3 `Last-Modified`) | Admissible? | Required leads |
|---|---|---|---|
| GFS 00Z of D-1 | **03:41 UTC** (grade A) | ✅ | +15 h … +38 h |
| GEFS 00Z of D-1 | **03:51 UTC** (f018) / **04:04 UTC** (f048) (grade A) | ✅ | +15 h … +38 h |
| ECMWF IFS 00Z of D-1 | **07:34 UTC** (grade A) | ❌ 2 h 34 m too late | — |
| ECMWF IFS 12Z of D-2 | **19:34 UTC** on D-2 (grade A) | ✅ | +27 h … +50 h |

Raw measurements (grade A, this session):

```
HEAD https://noaa-gefs-pds.s3.amazonaws.com/gefs.20260801/00/atmos/pgrb2bp5/gec00.t00z.pgrb2b.0p50.f018
  -> 200, Last-Modified: Sat, 01 Aug 2026 03:51:35 GMT, 96016246 bytes
HEAD .../gefs.20260801/00/atmos/pgrb2bp5/gec00.t00z.pgrb2b.0p50.f048
  -> 200, Last-Modified: Sat, 01 Aug 2026 04:04:08 GMT
HEAD https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20260801/00/atmos/gfs.t00z.pgrb2.0p25.f024
  -> 200, Last-Modified: Sat, 01 Aug 2026 03:41:15 GMT
HEAD https://ecmwf-forecasts.s3.amazonaws.com/20260801/00z/ifs/0p25/oper/20260801000000-24h-oper-fc.grib2
  -> 200, Last-Modified: Sat, 01 Aug 2026 07:34:06 GMT
HEAD https://ecmwf-forecasts.s3.amazonaws.com/20260801/12z/ifs/0p25/oper/20260801120000-36h-oper-fc.grib2
  -> 200, Last-Modified: Sat, 01 Aug 2026 19:34:08 GMT
```

Corroborating (grade B, ECMWF Confluence "Dissemination schedule", read in full):

> "Set I - Control forecast (formerly HRES) and Analysis Data (15-days) … 00 UTC | 0 to 90 by 1 |
> **05:45 → 06:12** … 150 to 360 by 6 | 06:27 → **07:34** … 12 UTC | … 17:45 → 18:12 … **18:27 → 19:34**"
> — https://confluence.ecmwf.int/display/DAC/Dissemination+schedule

and (grade B, ECMWF open-data page):

> "IFS data are released **at the end of the** real-time dissemination schedule. AIFS data are released
> as soon the data are produced." — https://www.ecmwf.int/en/forecasts/datasets/open-data

**Consequence:** any ECMWF-based feature must be built from the **D-2 12Z** cycle, i.e. ~13 h older
than the GFS/GEFS cycle we can use. That is a real skill handicap for ECMWF in *this* competition.

---

## 3. Question 1 — NOAA GEFS on `s3://noaa-gefs-pds`

### 3.1 Earliest archived date — grade A

```
GET https://noaa-gefs-pds.s3.amazonaws.com/?list-type=2&delimiter=/&max-keys=1000
  -> <CommonPrefixes><Prefix>gefs.20170101/</Prefix></CommonPrefixes>   (first prefix returned)
```
**Earliest date: `gefs.20170101/` (2017-01-01).**
Day-count check (grade A): `prefix=gefs.2023` → **365** day prefixes; `prefix=gefs.2024` → **366**.
**No missing days in either year.**

### 3.2 Directory layout — grade A

```
gefs.20230601/                -> 00/ 06/ 12/ 18/
gefs.20230601/00/             -> atmos/ chem/ wave/
gefs.20230601/00/atmos/       -> bufr/ init/ pgrb2ap5/ pgrb2bp5/ pgrb2sp25/
```

### 3.3 Variables / levels — grade A (literal `.idx` inventories)

`pgrb2sp25` (**0.25°**, "primary small" set) — full inventory of 38 records, wind entries only:
```
15:6363166:d=2023060100:UGRD:10 m above ground:3 hour fcst:ens mean
16:7192166:d=2023060100:VGRD:10 m above ground:3 hour fcst:ens mean
```
→ **the 0.25° GEFS product carries 10 m wind only. There is no 80 m or 100 m wind at 0.25°.**

`pgrb2bp5` (**0.5°**, "secondary" set), file `gep01.t00z.pgrb2b.0p50.f018.idx`:
```
354:70914397:...:UGRD:80 m above ground:18 hour fcst:ENS=+1     (record size 265.7 KB)
355:71186431:...:VGRD:80 m above ground:18 hour fcst:ENS=+1     (266.2 KB)
357:71602014:...:UGRD:100 m above ground:18 hour fcst:ENS=+1    (266.0 KB)
358:71874385:...:VGRD:100 m above ground:18 hour fcst:ENS=+1    (267.0 KB)
```
→ **80 m and 100 m u/v exist, but only in the 0.5° `pgrb2b` stream.**

`pgrb2ap5` (0.5° primary): 10 m only (`67:…UGRD:10 m above ground`, `68:…VGRD…`).

Spot-check that 100 m persists across both target years (grade A):
`gep15.t00z.pgrb2b.0p50.f024.idx` contains `UGRD:100 m above ground` on **20230101, 20230701,
20240101, 20240630, 20241231** — all `True`.

### 3.4 Resolutions and cadence — grade A

* horizontal: **0.25°** (`pgrb2s`, 10 m wind only) and **0.5°** (`pgrb2a`/`pgrb2b`, incl. 100 m wind)
* temporal: `gefs.20230601/00/atmos/pgrb2bp5/` contains **181 forecast hours** — `f000, f003, …, f240`
  (3-hourly) then `f246 … f840` (6-hourly, 00Z cycle only)
* members: **31** (`gec00` + `gep01 … gep30`), verified by exhaustive paginated listing (11 222 keys)
* one full 00Z `pgrb2bp5` cycle = **559.9 GB**
* cycles: 00/06/12/18 UTC (only 00Z runs to 840 h)
* ⚠ **no `geavg`/`gespr` in `pgrb2bp5`** — the ensemble mean/spread products exist only for the
  `pgrb2a`/`pgrb2s` primary sets, which have no 100 m wind. **Spread at 100 m must be computed from
  the 31 member files yourself.**

### 3.5 Licence — grade B (AWS Registry of Open Data page, read in full)

> "**License** — NOAA data disseminated through NODD are open to the public and can be used as
> desired. NOAA makes data openly available to ensure maximum use of our data … NOAA requests
> attribution for the use or dissemination of unaltered NOAA data. However, it is not permissible to
> state or imply endorsement by or affiliation with NOAA."
> — https://registry.opendata.aws/noaa-gefs/

> "**AWS CLI Access (No AWS account required)** — `aws s3 ls --no-sign-request s3://noaa-gefs-pds/`"

**Verdict: commercial use permitted, attribution requested. Passes the competition licence gate.**

### 3.6 Download size for a Korean box, 2 years of D-1 forecasts — grade D on grade-A measurements

**Important:** a GRIB2 record is a *global* field. A 4°×4° Korean box **does not** reduce the bytes
transferred when you byte-range a record out of `s3://noaa-gefs-pds`; you always pull the whole
global record and crop locally. (NOMADS `gribfilter` can subset spatially but keeps only ~10 days.)

Measured record size at 0.5°: **266 KB** per (variable, member, step).
Needed valid times per target day at 3-hourly: 15/18/21 UTC on D-1 and 00/03/06/09/12 UTC on D = **8 steps**
(from the D-1 00Z run: `f015 … f036`). Two variables (`UGRD/VGRD` at 100 m). 731 days (2023+2024).

| Ensemble subset | per day | 2023+2024 total |
|---|---:|---:|
| control `gec00` only | 4.2 MB | **3.0 GB** |
| 5 members | 20.8 MB | 14.8 GB |
| all 31 members (needed for spread) | 128.8 MB | **92.0 GB** |

Add 80 m u/v as well → ×2. Add 10 m at 0.25° (`pgrb2s`, ens-mean only) → negligible.

**Cheaper alternative (strongly preferred): the dynamical.org Zarr mirror** — see §5.1. It stores all
31 members in one chunk and needs only **~6–12 GB** for the same content (grade D estimate from the
grade-A chunk geometry).

### 3.7 Access recipe

```bash
# 1. fetch the GRIB index for the member/step you need (a few kB)
curl -s https://noaa-gefs-pds.s3.amazonaws.com/gefs.20230601/00/atmos/pgrb2bp5/gep07.t00z.pgrb2b.0p50.f024.idx
# 2. locate the two 100 m records and their byte offsets, then range-GET only those bytes
curl -s -r 71602014-72141374 \
  https://noaa-gefs-pds.s3.amazonaws.com/gefs.20230601/00/atmos/pgrb2bp5/gep07.t00z.pgrb2b.0p50.f024 \
  -o u100_v100.grib2
# 3. crop to 34-38N/126-130E locally (wgrib2 / cfgrib)
```
Pattern: `s3://noaa-gefs-pds/gefs.{YYYYMMDD}/{HH}/atmos/pgrb2bp5/{gec00|gepNN}.t{HH}z.pgrb2b.0p50.f{FFF}`

---

## 4. Question 2 — ECMWF open data `s3://ecmwf-forecasts`: does the archive begin in 2024?

### **REFUTED.** The archive begins **2023-01-18**, not 2024. — grade A

```
GET https://ecmwf-forecasts.s3.amazonaws.com/?list-type=2&delimiter=/&max-keys=1000
  -> 20230118/, 20230119/, 20230120/, …
```
Exhaustive month-by-month listing (grade A):
* **2023: 342 day-prefixes**, first `20230118/`, last `20231231/`.
  **Missing days: `20230427, 20230428, 20230429, 20230430, 20230501, 20230502`** (a 6-day hole) plus
  1 Jan – 17 Jan absent.
* **2024: 366 day-prefixes, complete.**

### 4.1 But the 2023 archive is the wrong product for wind power — grade A

2023 layout: `20230601/00z/0p4-beta/{enfo,oper,waef,wave}/` — **`0p4-beta`, i.e. 0.4°**.
2024 layout from 2024-02-01: `.../00z/0p25/…`, and from 2024-03-01: `.../00z/{aifs,ifs}/…`.
Binary-searched transitions (grade A): **first `0p25` day = 2024-02-01**; **first `aifs` day = 2024-02-29**.

Literal parameter inventory of the 2023 0.4° deterministic run
(`20230601/00z/0p4-beta/oper/20230601000000-24h-oper-fc.index`, 83 records, 52.4 MB):

```
surface: 10u, 10v, 2t, msl, sp, skt, st, ro, lsm, tcwv, tp
pressure levels (50,200,250,300,500,700,850,925,1000 hPa): d, gh, q, r, t, u, v, vo
```
→ **there is NO 100 m wind in the 2023 ECMWF open-data set.** Nearest hub-height proxies are 10 m and
u/v at 925/1000 hPa.

The 2024 0.25° set *does* have it (`20240601/12z/ifs/0p25/oper/…-36h-…index`, grade A):
```
100u  1.412 MB    100v  1.418 MB    10u  0.873 MB    10v  0.867 MB
```

**This is the killer practical objection to ECMWF here:** the dev year (2023) and the test year (2024)
would carry *different* feature definitions and *different* grid resolutions. Any 100 m-based ECMWF
feature cannot be validated on 2023 at all. To keep a constant definition you must use 10 m + 925 hPa
u/v on both years, throwing away the resolution and the hub-height level in 2024.

### 4.2 Steps, ensemble, and volume — grade A

2023 `oper` steps: `0,3,…,144` then `150,…,240` (65 steps); one 00Z `oper` run = **3.40 GB**.
2023 `enfo` (ensemble) 00Z run = **228 GB**, 177 objects; the 36 h index shows **50 perturbed members**
with `10u, 10v` and `u/v` at 200/500/700/850/925 hPa (`type=pf`). No `em`/`es` (mean/spread) products.

Extraction cost (grade D on grade-A record sizes), D-2 12Z run, 8 steps/day, 731 days:
* deterministic, 2023 0.4°, {10u,10v,u925,v925,u1000,v1000} = 6 records = 3.65 MB/step → **21 GB**
* deterministic, 2024 0.25°, {100u,100v,10u,10v} = 4.57 MB/step → **13.1 GB** for 2024 alone
* ensemble (51 members × 10u/10v, 0.4°) ≈ 514 MB/day → **≈ 376 GB**. Not practical.

### 4.3 Licence — grade B (ECMWF open-data page, read in full)

> "A subset of ECMWF real-time forecast data from the IFS and AIFS models is made available to the
> public free of charge. Their use is governed by the **Creative Commons CC-BY-4.0 licence** and the
> ECMWF Terms of Use. **This means that the data may be redistributed and used commercially**, subject
> to appropriate attribution."
> — https://www.ecmwf.int/en/forecasts/datasets/open-data

> "**Rolling archive and data availability:** ECMWF Open Data provides access to real-time forecast
> data on a rolling archive basis. Data are retained for the **most recent 12 forecast runs**,
> corresponding to approximately 2–3 days of forecasts …" *(this is the ECMWF portal; the AWS mirror
> `s3://ecmwf-forecasts` is what actually holds the 2023→ archive)*

Access pattern:
`s3://ecmwf-forecasts/{YYYYMMDD}/{HH}z/0p4-beta/oper/{YYYYMMDDHHMMSS}-{S}h-oper-fc.grib2` (2023)
`s3://ecmwf-forecasts/{YYYYMMDD}/{HH}z/ifs/0p25/oper/{YYYYMMDDHHMMSS}-{S}h-oper-fc.grib2` (2024-03→)
Each `.grib2` has a sibling `.index` (one JSON object per record with `_offset`/`_length`) → clean
byte-range subsetting.

---

## 5. Question 3 — every other candidate checked

### 5.1 ✅ dynamical.org "NOAA GEFS forecast, 35 day" (Zarr mirror of GEFS) — **PASSES, and is the best access path**

Grade B (catalog page read in full) + grade A (zarr metadata fetched):

> "**Time domain** — Forecasts initialized **2020-10-01 00:00:00 UTC to Present** … Forecasts initialized
> every 24 hours … **Forecast step 0-240 hours: 3 hourly** … ensemble_member 0–30 …
> **Dataset licensed under CC BY 4.0.** Attribution … 'NOAA NWS NCEP GEFS data processed by
> dynamical.org from NOAA Open Data Dissemination archives.'"
> — https://dynamical.org/catalog/noaa-gefs-forecast-35-day/ and https://source.coop/dynamical/noaa-gefs-forecast-35-day

> "**Interpolation** — Source data is available at both 0.25-degree and 0.5-degree resolutions. All
> variables except the 100m wind components are derived from a 0.25-degree grid for the first 240 hours
> … **100m wind components are derived from a 0.5-degree grid for all lead times.** Bilinear
> interpolation is used to convert 0.5-degree data to a 0.25-degree grid."

Variables include `wind_u_100m`, `wind_v_100m`, `wind_u_80m`, `wind_v_80m`, `wind_u_10m`, `wind_v_10m`,
`temperature_80m`, `pressure_80m`, `wind_gust_surface` (grade B, variable table).

Grade A — array geometry actually fetched from
`https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr/wind_u_100m/zarr.json`:
```json
{"shape":[2137,31,181,721,1440],
 "chunk_grid":{"configuration":{"chunk_shape":[1,31,192,374,368]}},
 "codecs":[{"name":"sharding_indexed","configuration":{"chunk_shape":[1,31,64,17,16], …}}],
 "dimension_names":["init_time","ensemble_member","lead_time","latitude","longitude"],
 "attributes":{"long_name":"100 metre U wind component","units":"m s-1"}}
```
Inner chunk = `(1 init, 31 members, 64 leads, 17 lat, 16 lon)` = 2.16 MB uncompressed (zstd-3 on disk).
A 34–38 N / 126–130 E box is 17×17 points at 0.25° → **≈ 2×2 = 4 inner chunks per (init, variable)**;
one lead chunk (64 steps ≈ 8 days) covers all needed leads.
→ **≈ 8–17 MB per init for u100+v100 including all 31 members; ~6–12 GB for 2023+2024** (grade D).

Access recipe:
```python
import xarray as xr
ds = xr.open_zarr("https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr?email=optional@email.com")
box = ds[["wind_u_100m","wind_v_100m"]].sel(latitude=slice(38,34), longitude=slice(126,130))
```
⚠ **Caveat (grade A):** `init_time` in this dataset is **00 UTC only** ("This dataset contains only the
00 hour UTC initialization times which produce the full-length, 35-day forecast"). That is exactly the
cycle we want (D-1 00Z), so this is not a limitation here — but it means no 12Z fallback.
⚠ Not verified: whether the whole 2023-01-01 → 2024-12-31 init range is gap-free in this mirror.
`shape[0]=2137` inits from 2020-10-01 to "present" is consistent with daily, but I did **not** read the
`init_time` coordinate values. `⚠ NOT VERIFIED`.

### 5.2 ❌ NCEP GFS on `s3://noaa-gfs-bdp-pds` — passes every gate but has **zero decorrelation value**

Grade A: earliest prefix `gfs.20210101/`; 2023 and 2024 present. GFS **is already one of the two
competition-supplied sources**, so adding it contributes nothing. Listed only for completeness.

### 5.3 ❌ DWD ICON via `opendata.dwd.de` — **no archive**

Grade B (two independent full-text reads):

> "Note: **The raw files are deleted after 24 hours, and there is no long-term archive available
> publicly.**" — https://huggingface.co/datasets/openclimatefix/dwd-icon-global (dataset card)

> "**DWD maintains only a 24-hour rolling window of files on their servers**, some of which we archive
> here." — https://source.coop/dynamical/dwd-icon-grib (README)

DWD's own licence is fine (grade B, quoted in the dynamical README):
> "**Data license: Creative Commons BY 4.0.** To quote DWD's legal notice: 'All freely accessible
> geodata and geodata services … may be reused under the terms of the Creative Com[mons]…'"

But there is no DWD-operated archive of past runs. Two third-party archives exist:

#### 5.3a ⚠ Open Climate Fix `openclimatefix/dwd-icon-global` on HuggingFace — **covers Mar 2023 →, but impractical**

Grade B (dataset card) + grade A (HF tree API):
> "This dataset is comprised of forecasts from the German Weather Service's (DWD) ICON-Global model
> **from March 2023 to the present with all variables included**. Each forecast runs up to 4 days into
> the future, and the model is ran 4 times per day … **License: cc-by-4.0**"

```
GET https://huggingface.co/api/datasets/openclimatefix/dwd-icon-global/tree/main/data
  -> data/2023, data/2024, data/2025
GET .../tree/main/data/2023   -> 3,4,5,…,12   (March 2023 onward — no Jan/Feb 2023)
GET .../tree/main/data/2023/6/1
  -> 20230601_00.zarr.zip  42,108,883,861 bytes
     20230601_06.zarr.zip  42,134,372,512 bytes  …
```
**42 GB per run × 4 runs/day × ~640 days ≈ 100 TB.** Random access into `zarr.zip` over HTTPS is
theoretically possible, but the dataset is **gated** ("you have to accept the conditions to access its
files") and the practical engineering cost is far above the other candidates.
⚠ NOT VERIFIED: whether ICON-Global's variable set inside these zips includes a hub-height wind
(ICON global's single-level output is 10 m; 100 m would have to come from model levels).

#### 5.3b ❌ dynamical.org `dwd-icon-grib` — ICON-**EU** only (Europe domain), created 2025-10-08. Useless for Korea.

### 5.4 ❌ Météo-France ARPEGE/AROME — **14-day rolling archive, and no Korea-relevant product**

Grade A: `s3://mf-nwp-models/` top level = `arome-france-hd/, arome-france/, arpege-europe/,
arpege-world/`. Exhaustive key listing of `arpege-world/` and `arpege-europe/` returns **only**
`static/landmask.grib2` and `static/terrain.grib2` — **no dated forecast data at all in the bucket
right now**.

Grade C (⚠ snippet only, data.gouv.fr organisation page):
> "Les paquets de données des Prévisions Numériques du Temps sont **archivés 14 jours** et mis à
> disposition sur une plateforme dédiée" — https://www.data.gouv.fr/organizations/meteo-france

AROME covers France only; ARPEGE-world at 0.25° would cover Korea but there is no 2023–2024 archive.
**Fails the archive gate.**

### 5.5 ❌ Environment and Climate Change Canada (GDPS/GEPS) — **no free archive**

Grade B (MSC Open Data FAQ, full page read):
> "**Environment and Climate Change Canada does not have an online service to retrieve archived data.**
> The data retrieval service from our archive is under a cost recovery policy. **We charge 118$/hour,
> with a minimum charge of 118$**, to retrieve/recreate the requested data."
> — https://eccc-msc.github.io/open-data/faq/readme_en/

CaSPAr (Univ. of Waterloo) exists as an academic mirror (grade C, snippet only — ⚠ NOT VERIFIED for
licence, domain coverage or global GDPS content). ECCC's own Datamart is rolling-window only.
The only practical 2023 path to GEM is Open-Meteo (§5.7), which is leakage-contaminated.

### 5.6 ⚠ JMA MSM / GSM via RISH Kyoto — **covers Korea, archive is deep, but the licence fails**

The JMA Meso-Scale Model (MSM) domain is **120–150 E, 22.4–47.6 N** (grade A, from Open-Meteo's
`meta.json`: `BBOX[22.4,120.0,47.6,150.0]`, `temporal_resolution_seconds: 3600`, 0.05° ≈ 5 km) — it
**does** cover the Korean peninsula, hourly, at 5 km. On paper this is the single most attractive
*meteorological* candidate: a genuinely independent regional model at higher resolution than LDAPS'
neighbours.

But the RISH Kyoto archive states its own terms (grade B, full page read, ISO-2022-JP decoded,
http://database.rish.kyoto-u.ac.jp/arch/jmadata/):

> 「（財）気象業務支援センターを通して公開されている気象庁作成の数値予報データ・観測データのダウンロードサイトです．
> **ここでは教育研究機関向けにデータを提供しています．企業活動等のためにデータを頻繁に必要とされる方は，
> 気象業務支援センターからデータを直接購入し**，データ提供スキーム全体の維持発展にご協力ください．」

("This is a download site for JMA numerical-prediction and observation data released through the
Japan Meteorological Business Support Center (JMBSC). **Data here is provided for educational and
research institutions. Those who need the data frequently for corporate activities should purchase
the data directly from JMBSC.**")

**Verdict: the RISH mirror is explicitly scoped to education/research, with commercial use routed to a
paid JMBSC subscription. It does not satisfy "licensed for commercial use". HIGH legal risk — do not
use without an explicit ruling.**
⚠ NOT VERIFIED: JMA's own site terms for GPV redistribution; JMBSC pricing.

### 5.7 ❌ Open-Meteo — **licence is fine, but the historical products are either leaky or too short**

This one needs care because it looks perfect and is not.

**(a) The bulk S3 bucket `s3://openmeteo` is genuinely open (grade B, AWS Registry, read in full):**
> "**License** — CC-BY 4.0 … `aws s3 ls --no-sign-request s3://openmeteo/`"
> — https://registry.opendata.aws/open-meteo/

**(b) But `data/` is a *stitched* series, not a forecast at fixed lead (grade B, docs read in full):**
> "**Historical Forecast API:** A continuous hourly timeseries built by **stitching the first hours of
> each successive model run**. Closely tracks actual conditions because each run is initialised from
> real measurements. Coverage starts around 2021." — https://open-meteo.com/en/docs/historical-forecast-api

Stitching the *first hours* of each run means the value stored for valid time `T` comes from a run
initialised only a few hours before `T`. For `T` on day D that run is initialised **after** the
D-1 05:00 UTC basis time. **Using `data/` as a day-ahead feature is straightforward target leakage.**
This is a fatal, not a cosmetic, objection.

**(c) The leak-free per-run stores exist but only for 2026 (grade A):**
```
data_run/dwd_icon/     -> ['data_run/dwd_icon/2026/']
data_spatial/jma_msm/  -> ['data_spatial/jma_msm/2026/']
```

**(d) The `Previous Runs API` IS leak-free — and JMA MSM reaches back into 2022 (grade B + grade A):**
> "Data from past model runs is aligned to **fixed lead-time offsets of 1–7 days**. Requesting
> `temperature_2m_previous_day1` returns **the value predicted 24 hours before valid time**;
> `_previous_day2` returns 48 hours before … Most models are **archived from January 2024**. GFS 2 m
> temperature extends back to March 2021." — https://open-meteo.com/en/docs/previous-runs-api
> and (historical-forecast page): "Data starts from January 2024 (**GFS from March 2021, JMA from 2018**)."

Grade A probes I ran (`previous-runs-api.open-meteo.com`, point 36.0 N / 126.5 E, `wind_speed_10m_*`):

| model | date | `_previous_day1` | `_previous_day2` |
|---|---|---|---|
| `jma_msm` | 2022-12-01 | 24/24 values | 24/24 values |
| `jma_msm` | 2023-01-01 | 24/24 | 24/24 |
| `jma_msm` | 2023-06-01 | 24/24 | 24/24 |
| `jma_msm` | 2024-06-01 | 24/24 | 24/24 |
| `jma_msm` | 2024-12-31 | 24/24 | 24/24 |
| `jma_msm` | 2018-06-01 | 0/24 | 0/24 |
| `icon_global` | 2023-06-01 | 0/24 | 0/24 |
| `icon_global` | 2024-06-01 | 24/24 | 24/24 |
| `gfs_global`/`gfs025`/`gfs_seamless` | 2023-06-01 | 0/24 | 0/24 |

So **only `jma_msm` has a leak-free, fixed-lead archive spanning both 2023 and 2024.**
Lead-time admissibility (grade D): with basis D-1 05:00 UTC and targets D-1 15:00 → D 14:00 UTC, the
required lead ranges from **10 h to 33 h**. `_previous_day1` = 24 h lead is admissible only for target
hours up to 14:00 KST and **leaks for 15:00–23:00 KST**; `_previous_day2` = 48 h lead is admissible for
every hour but is a materially weaker forecast.

**(e) Two blockers remain even for `jma_msm` previous-runs:**
1. **Variables.** Open-Meteo's JMA MSM store carries only
   `wind_u_component_10m`, `wind_v_component_10m`, `wind_gusts`, `temperature_2m`, `pressure_msl`,
   `relative_humidity_2m`, cloud, precipitation, `shortwave_radiation` (grade A, prefix listing of
   `data/jma_msm/`). **No 80 m or 100 m wind.** Hub height would need shear extrapolation.
2. **Terms of the API route.** Grade B (https://open-meteo.com/en/terms, read in full):
   > "**Non-Commercial Use** — By using the Free API for non-commercial use you agree to following terms:
   > Less than 10'000 API calls per day, 5'000 per hour and 600 per minute. **You may only use the free
   > API services for non-commercial purposes.** You accept to the CC-BY 4.0 licence…"

   The *data* is CC-BY-4.0 (commercial OK); the *free API service* is contractually non-commercial and
   rate-limited. And the previous-runs archive is **not** in the CC-BY-4.0 S3 bucket — only `data/`
   (the leaky one) is. So the only leak-free path runs through the restricted API.
   Also note the project rule "remote API **inference** is FORBIDDEN": fetching archived numbers is
   arguably retrieval rather than inference, but this is a judgement call, not a settled reading.

**Verdict: Open-Meteo fails.** The open-licence bulk path is leaky; the leak-free path is behind a
non-commercial, rate-limited API and lacks hub-height wind.

*(Incidental grade-A findings from the bucket, useful if the leak objection is ever lifted for a
non-day-ahead use: `data/dwd_icon/wind_u_component_80m` chunks span **2023-05-25 → present** (112
contiguous chunks, 45.1 GB for that one variable); `data/jma_msm/wind_u_component_10m` spans
**2023-05-12 → present** (250 chunks, 3.5 GB); `data/ukmo_global_deterministic_10km/wind_speed_10m`
spans **2022-02-27 → present** (203 chunks, 91.0 GB); `data/cmc_gem_gdps/wind_speed_80m` spans
**2023-05-20 →**; `data/ecmwf_ifs025/wind_u_component_100m` only from **2024-03-04**. Chunk→time
arithmetic is grade D on grade-A chunk indices and grade-A `meta.json` (`chunk_time_length`,
`temporal_resolution_seconds`).)*

### 5.8 ⚠ NOAA AIWP / MLWP `s3://noaa-oar-mlwp-data` — deep archive with **native 100 m wind**, but rule-risky

Grade B (bucket `README.txt` and AWS Registry, both read in full):
> "This is an archive of pure AI-based numerical weather prediction **reforecasts** … The current period
> of record is roughly **09/30/20 to present** … 00 UTC and 12 UTC initializations are available for all
> three years. 06 UTC and 18 UTC initializations are available for 2023 only."
> "**FourCastNet v2-small**: 10-m u wind component, 10-m v wind component, **100-m u wind component,
> 100-m v wind component**, 2-m temperature, Surface pressure, …"
> "**License** — Open Data. There are no restrictions on the use of this data."
> — https://registry.opendata.aws/aiwp/ ; s3://noaa-oar-mlwp-data/README.txt

Grade A coverage counts: `FOUR_v200_GFS/2023/` = **365** day prefixes, `/2024/` = **365**;
`PANG_v100_GFS` 365/364; `GRAP_v100_GFS` 365/366.
Grade A file sizes: `FOUR_v200_GFS_2023060100_f000_f240_06.nc` = **7.50 GB** (0.25°, 41 steps × 6 h).
Grade A: a **Zarr/kerchunk view** exists at `parquet/FOUR_v200_GFS_combined_all.parq/` with arrays
`u100`, `v100`, `u10`, `v10`, `t2`, `sp`, `msl`, `tcwv`, `u`, `v`, `t`, `z`, `r`:
```
u100/.zarray  shape [3973, 41, 721, 1440]  chunks [1, 1, 721, 1440]  dtype <f4  compressor null
```
→ one chunk = one global field = 4.15 MB, **uncompressed**; a Korean box costs a full global chunk.
For 4 valid times/day × 2 vars × 731 days ≈ **24 GB** (grade D).

**Three reasons this ranks below GEFS despite the 100 m wind:**
1. **6-hourly output only** (`ZZ = 06`), against an hourly target. Interpolation across 6 h destroys the
   sub-daily wind structure that drives the settlement bands.
2. **These are retrospective reforecasts, not real-time products.** The AWS Registry calls them
   "reforecasts"; the 2020–2023 portion was generated after the fact. A strict reading of "every
   forecast input must have been available by the official basis time" fails on file provenance even
   though the *information content* (a GFS analysis at 00/12 UTC) respects the cut-off. **⚠ Judgement
   call — needs an explicit ruling before use.**
3. **Upstream weight licences conflict with the NOAA "no restrictions" statement.** Grade C
   (snippets only, ⚠ NOT VERIFIED from full text):
   > "The trained parameters of Pangu-Weather were made available under the terms of the **BY-NC-SA 4.0**
   > license. **The commercial use of these models is forbidden.**" — github.com/198808xc/Pangu-Weather
   > "The model weights are made available for use under the terms of the Creative Commons
   > **Attribution-NonCommercial-ShareAlike 4.0** International (CC BY-NC-SA 4.0)."
   > — github.com/google-deepmind/weathernext
   FourCastNet is the exception (grade C): "This model is ready for **commercial use**. This model is
   licensed under the NVIDIA AI Product Agreement." — build.nvidia.com/nvidia/fourcastnet/modelcard.
   **So if the AIWP archive is used at all, use only `FOUR_v200_GFS` (which is also the only member
   with 100 m wind) and never `PANG_*` or `GRAP_*`.**

### 5.9 ❌ Running GraphCast / Pangu open weights locally

Blocked at the licence gate: BY-NC-SA 4.0 forbids commercial use (grade C snippets above; the project
rule requires "licensed for commercial use"). Independently, admissible initial conditions **do** exist
(GFS analysis `gfs.tHHz.pgrb2.0p25.anl` in `s3://noaa-gfs-bdp-pds` from 2021-01-01, grade A) so the
*data* side is solvable — but it is moot while the weights are NC. FourCastNet v2-small under the
NVIDIA agreement is the only commercially usable option, and NOAA has already run it for us (§5.8),
so local inference adds nothing but risk.
⚠ NOT VERIFIED: full text of the NVIDIA AI Product Agreement.

### 5.10 ❌ Met Office global 10 km `s3://met-office-atmospheric-model-data`

Grade A:
```
GET https://met-office-atmospheric-model-data.s3.amazonaws.com/?prefix=global-deterministic-10km/&delimiter=/
  -> 20240805T1800Z/, 20240806T0000Z/, …
```
**Earliest run = 2024-08-05 18Z.** No 2023, and only 5 months of 2024. Fails the archive gate.
(Open-Meteo's `ukmo_global_deterministic_10km` store goes back to 2022-02-27 (grade A/D) but is the
leaky stitched series, §5.7b, and has no 80/100 m wind.)

### 5.11 ❌ KMA (기상자료개방포털 / API Hub)

Grade B (full page read, https://data.kma.go.kr/data/rmt/rmtList.do?code=340&pgmNo=65 — LDAPS; and
`code=312` — RDAPS):
> "단일면 주요 기상요소 20종에 대한 파일셋 구성하여 '경량화' 탭 통해 제공(제공기간/방법: **2021.1.~**/웹 다운로드)
>  * 1회 최대 신청용량 100GB
>  * **최근 1년 자료까지만 제공**"
> "※ 2026년 3월 중 **통합모델(UM) 자료가 제공 중단 예정**입니다."

("The lightweight fileset covers 2021.1 onward, **but only the most recent 1 year of data is
provided**"; "UM model data provision is scheduled to be **discontinued** during March 2026.")

Independently: KMA LDAPS **is** the competition's own source, so decorrelation is ~0 by construction.
KIM (Korean Integrated Model) via API Hub is a distinct model, but ⚠ NOT VERIFIED whether any 2023
archive is retrievable, and the same 1-year retention likely applies.

### 5.12 ❌ Reanalysis products (ERA5, MERRA-2, ERA5-Land, `copernicus_era5*` in the Open-Meteo bucket)

Forbidden by rule. Noted only because `s3://openmeteo/data/copernicus_era5*` is sitting right next to
the forecast stores and is easy to grab by accident. **Do not.**

---

## 6. Question 4 — passing candidates, consolidated

| # | Candidate | Archive start | Covers 2023 | Covers 2024 | Res. | Cadence | Hub-height wind | Licence | Access recipe |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **NOAA GEFS** (`s3://noaa-gefs-pds`) | **2017-01-01** | ✅ 365 d | ✅ 366 d | 0.5° (100 m); 0.25° (10 m) | 3-hourly to f240; 4 cycles/day; 31 members | **`UGRD/VGRD:100 m above ground`** and `:80 m` in `pgrb2b` | NODD open, commercial OK, attribution requested | `s3://noaa-gefs-pds/gefs.{YYYYMMDD}/00/atmos/pgrb2bp5/{gec00\|gepNN}.t00z.pgrb2b.0p50.f{FFF}` + `.idx` byte-range |
| 1b | **same, via dynamical.org Zarr** | 2020-10-01 | ✅ | ✅ | 0.25° grid (100 m bilinearly upsampled from 0.5°) | 3-hourly, 00Z inits only, 31 members | `wind_u_100m`, `wind_v_100m`, `wind_u_80m`, `wind_v_80m` | **CC BY 4.0** | `xr.open_zarr("https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr")` |
| 2 | **ECMWF open data IFS** (`s3://ecmwf-forecasts`) | **2023-01-18** (6-day hole in Apr/May 2023) | ⚠ 342 d | ✅ 366 d | 0.4° in 2023 → 0.25° from 2024-02-01 | 3-hourly to 144 h; 4 cycles/day | ❌ 2023 (10 m + u/v 925/1000 hPa only) · ✅ `100u/100v` from 2024-02-01 | **CC-BY-4.0, commercial OK** | `s3://ecmwf-forecasts/{YYYYMMDD}/12z/0p4-beta\|ifs/0p25/oper/…-{S}h-oper-fc.grib2` + `.index` byte-range |
| 3 | **NOAA AIWP FourCastNetv2** (`s3://noaa-oar-mlwp-data/FOUR_v200_GFS/`) | 2020-09-30 | ✅ 365 d | ✅ 365 d | 0.25° | **6-hourly** to 240 h; 00/06/12/18Z in 2023, 00/12Z in 2024 | ✅ native `u100`/`v100` | "Open Data. There are no restrictions" — but see §5.8 note 3 | `s3://noaa-oar-mlwp-data/FOUR_v200_GFS/{YYYY}/{mmdd}/FOUR_v200_GFS_{YYYYmmddhh}_f000_f240_06.nc`, or the Zarr view `parquet/FOUR_v200_GFS_combined_all.parq/{u100,v100}` |

Everything else in §5 fails at least one gate.

---

## 7. Question 5 — ranking by (expected decorrelation) × (feasibility)

Decorrelation scores below are **priors, not measurements** (grade D). The repo has already measured
LDAPS↔GFS error correlation ≈ 0.78; I use model lineage (analysis system, dynamical core, resolution)
to place each candidate relative to that.

| Rank | Candidate | Expected decorrelation from LDAPS+GFS | Feasibility | Product | Verdict |
|---:|---|---|---|---|---|
| **1** | **GEFS ensemble *spread* (31 members, 100 m)** via dynamical.org Zarr | **Mean: LOW (~0.15)** — same GDAS analysis, same FV3 core as GFS. **Spread: HIGH (~0.8)** — a channel neither LDAPS nor GFS provides at all | **HIGH** — CC-BY-4.0, 6–12 GB, one `xr.open_zarr` call, D-1 00Z run lands 03:51 UTC | **HIGH** | ✅ **recommended** |
| 2 | ECMWF IFS deterministic, D-2 12Z | **MEDIUM-HIGH (~0.5)** — genuinely independent 4D-Var analysis and dynamical core; the single most skilful global model | **MEDIUM** — 21 GB (2023 spec) / 13 GB (2024 spec), byte-range plumbing, **but the 2023 and 2024 feature sets differ** and the usable run is 13 h staler | MEDIUM | ⚠ second, with a caveat that could kill it |
| 3 | AIWP FourCastNetv2 (100 m, GFS-initialised) | **MEDIUM (~0.4)** — same initial conditions as GFS, but a completely different (learned) propagator; errors decorrelate with lead time | **MEDIUM-LOW** — 6-hourly only; ~24 GB via global chunks; **retrospective reforecast provenance is a rule risk** | MEDIUM | ⚠ third, needs a rules ruling |
| 4 | JMA MSM 5 km hourly (Korea in domain) | **HIGH (~0.7)** — independent regional model, 5 km, hourly, and the only candidate whose native resolution beats the LDAPS neighbourhood | **LOW** — RISH mirror is education/research-scoped; Open-Meteo route is 10 m only + non-commercial API | **BLOCKED on licence** | ❌ |
| 5 | Open-Meteo bulk `data/` (ICON, GEM, UKMO, MSM) | HIGH on paper | **ZERO** — stitched-first-hours series ⇒ target leakage | — | ❌ |
| 6 | DWD ICON global via OCF HuggingFace | HIGH (~0.7) | **VERY LOW** — 42 GB/run × ~2 560 runs ≈ 100 TB, gated | — | ❌ |
| 7 | Météo-France / ECCC / Met Office / KMA archives | n/a | fail archive or cost gate | — | ❌ |

### Single top recommendation

> **Add GEFS as an uncertainty source, not as a second opinion on the mean.**
> Pull `wind_u_100m` / `wind_v_100m` (and `wind_u_80m` / `wind_v_80m`) for all 31 members from the
> **D-1 00Z** cycle over 34–38 N / 126–130 E, 2023-01-01 → 2024-12-31, via
> `https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr` (CC BY 4.0, ~6–12 GB).
> Derive per-site, per-hour features: **ensemble spread, inter-quartile range, member-wise
> exceedance probability of the group capacity bands, and the spread's lead-time growth rate.**
> Feed those to the *policy* layer (`T*_G*` threshold/gain selection), not to the point-forecast layer.

**Main risk (stated plainly).** GEFS is initialised from the same GDAS analysis and integrated with
the same FV3 core as the GFS block already in the model. Its *ensemble mean* will be more correlated
with GFS than LDAPS is (repo-measured 0.78), so the AGENTS.md "external NWP closed on evidence"
finding — LDAPS-GFS correlation ~0.78 caps averaging gain at 4.6 % against ~11.0 % required — applies
*a fortiori* to the GEFS mean. **The bet is entirely on the spread channel.** If a spread feature does
not move FICR on the fold-outside gate, the axis is dead and 92 GB (or 12 GB) of download was wasted.
Secondary risks: (i) GEFS 100 m wind is only 0.5° (~55 km), coarser than the LDAPS 1.5 km grid the
model already sees, so it cannot add terrain detail — only synoptic uncertainty; (ii) the dynamical.org
mirror is a third-party re-encoding whose init-time completeness over 2023–2024 I did **not** verify
(§5.1), so the first step must be to read the `init_time` coordinate and confirm 731 contiguous inits
before committing to the axis; (iii) the D-1 00Z cycle's 03:51 UTC arrival leaves only ~1 h of margin
before the 05:00 UTC basis — fine for a retrospective study, tight for any future live deployment.

**Explicit non-recommendation.** Do **not** use Open-Meteo's Historical Forecast API or its
`s3://openmeteo/data/` store for day-ahead features. It looks ideal — CC-BY-4.0, ICON/GEM/UKMO/MSM,
100 m and 80 m winds, back to 2022 — and it is leaky by construction. That is the single most likely
way this lane's findings could be misused.

---

## 8. Search log

29 web queries were issued via the `websearch` skill (budget was ~80).

| # | Query |
|---:|---|
| 1 | noaa-gefs-pds AWS Open Data Registry GEFS |
| 2 | GEFS pgrb2s p25 variables list 100 m wind UGRD 100 m above ground GEFS 0.25 degree |
| 3 | GEFS v12 output variables 80 m above ground UGRD VGRD pgrb2a pgrb2b inventory |
| 4 | ECMWF open data real-time forecasts licence CC-BY-4.0 Creative Commons Attribution 4.0 |
| 5 | ecmwf-forecasts AWS open data registry ECMWF real-time open data |
| 6 | noaa-oar-mlwp-data AWS open data registry machine learning weather prediction FourCastNet Pangu GraphCast |
| 7 | AWS open data registry Meteo-France ARPEGE AROME s3 bucket open data |
| 8 | Meteo-France open data ARPEGE archive meteo.data.gouv.fr historique archives 2023 |
| 9 | mf-models-on-aws.org documentation ARPEGE world archive retention how long data kept |
| 10 | Open-Meteo AWS open data registry s3 openmeteo bucket weather api database |
| 11 | Open-Meteo Historical Forecast API license terms CC BY 4.0 commercial use |
| 12 | ECMWF open data dissemination schedule availability time HRES 00 UTC real-time open data delay |
| 13 | NCEP production suite schedule GEFS 00Z availability time pgrb2b |
| 14 | Open-Meteo Historical Forecast API documentation data availability start date ERA5 reanalysis fill archive 2022 |
| 15 | 気象庁 数値予報GPV MSM 京都大学 RISH アーカイブ 利用条件 商用 気象業務支援センター |
| 16 | ECMWF real-time dissemination schedule HRES 00 UTC delivery complete 06:55 UTC ensemble 08:20 |
| 17 | opendata.dwd.de ICON numerical weather prediction data availability retention deleted after 24 hours archive |
| 18 | Pangu-Weather GitHub 198808xc license BY-NC-SA pretrained weights commercial use |
| 19 | GraphCast google-deepmind license CC BY-NC-SA 4.0 model weights commercial use prohibited |
| 20 | dynamical.org catalog noaa gefs forecast 35 day 100m wind variable data availability 2020 zarr |
| 21 | NCEP GEFS 00Z model run completion time production schedule available UTC pgrb2b 0.5 degree |
| 22 | NCEP production suite schedule GEFS "available" time chart nco pmb prodstat GEFS 00Z 05:00 UTC |
| 23 | KMA 기상청 API 허브 수치예보모델 GDPS LDPS 과거 자료 제공 기간 archive 2023 |
| 24 | FourCastNet v2 small license NVIDIA modulus weights terms commercial use ai-models fourcastnetv2 |
| 25 | CaSPAr archive Environment Canada GDPS HRDPS numerical weather prediction archive registration license |
| 26 | 気象庁 数値予報GPV 利用 著作権 二次利用 商用利用 可能 気象庁ホームページ利用規約 |
| 27 | ECMWF open data 2022 launch CC-BY-4.0 0.4 degree real-time forecasts free January 2022 announcement |
| 28 | 기상자료개방포털 수치모델 과거 자료 다운로드 LDAPS GDAPS GRIB 제공 기간 |
| 29 | KMA API Hub 수치모델 자료 제공 기간 최근 며칠 과거자료 신청 FTP |

### Direct HTTP/S3 probes (the grade-A evidence base)

S3 `ListObjectsV2` / `GET` / `HEAD` against, among others:
`noaa-gefs-pds`, `noaa-gfs-bdp-pds`, `ecmwf-forecasts`, `noaa-oar-mlwp-data`, `mf-nwp-models`,
`met-office-atmospheric-model-data`, `openmeteo`, `noaa-hrrr-bdp-pds`, `noaa-nws-graphcastgfs-pds`;
plus `data.dynamical.org` zarr metadata, `huggingface.co/api/datasets/...` tree listings,
`historical-forecast-api.open-meteo.com` and `previous-runs-api.open-meteo.com` point probes,
and full-text reads of registry.opendata.aws (noaa-gefs, aiwp, open-meteo), ecmwf.int, confluence.ecmwf.int,
open-meteo.com (docs/terms), eccc-msc.github.io, data.kma.go.kr, database.rish.kyoto-u.ac.jp,
source.coop, dynamical.org, huggingface.co. Full URLs are inline in §2–§5 with their responses quoted.

---

## 9. What is explicitly NOT established

* No skill or error-correlation number for any candidate against the competition's own LDAPS/GFS
  features. **Every decorrelation figure in §7 is a prior, not a measurement.**
* Whether the dynamical.org GEFS mirror has all 731 inits for 2023-01-01 → 2024-12-31 (`⚠ NOT VERIFIED`).
* Whether the OCF HuggingFace ICON-Global zips contain a hub-height wind (`⚠ NOT VERIFIED`).
* Full text of the NVIDIA AI Product Agreement (`⚠ NOT VERIFIED`).
* Whether the competition organisers would accept (a) a retrospectively generated reforecast archive,
  or (b) archived-value retrieval from a remote API, under their rules. **Both are judgement calls that
  the root session must settle before any download.**
* CaSPAr's licence, domain, and whether it carries GDPS globally (`⚠ NOT VERIFIED`, grade C only).
* No data was downloaded; no fit was run; nothing outside `research/` was written.
