# A Pipeline for CT-based Bone Fracture Point Cloud Reconstruction and Visualization

A retrospective reproduction of the data preparation pipeline from Yuan Qinhui's "Research on 3D Tibial Plateau Fracture Reduction Method." This project reconstructs bone surfaces from CT scans, simulates fracture fragmentation through multiple methods, and generates labeled point cloud data — all without deep learning, designed for CPU-only execution.

## Pipeline Overview

```
CT Volume (ct.nii.gz) → Marching Cubes Mesh (.ply) → Sampled Point Cloud (.npz, N=4096)
  → Fracture Simulation (3 methods) → Fracture Surface Labeling → Visualization
```

### Step 0 — Case Selection & Overview

Automatically selects diverse bone types from 102 TotalSegmentator cases. Each bone group is capped at 10 cases, prioritizing larger volumes. Outputs `selected_cases.json` (120 entries, 30 unique cases, 13 bone groups).

### Step 1 — Bone Surface Reconstruction

Reads CT volume + bone mask via nibabel, applies Marching Cubes (scikit-image) on the masked region, retains the largest connected component, and converts vertex coordinates to real-world millimeters using voxel spacing. Outputs triangular mesh `.ply` files.

- **Result**: 120/120 meshes generated in ~80s
- **Vertex range**: 2,088 ~ 142,600 (median ~26,206)

### Step 2 — Point Cloud Sampling

Uniform area-weighted sampling via `trimesh.sample.sample_surface` at N=4096 points per mesh. Normals are explicitly normalized. Point clouds are centered and unit-sphere scaled. Both normalized and raw (mm) coordinates are saved.

- **Result**: 120/120 sampled, ~0.4s per batch
- **Scale factor range**: 34.0 ~ 187.7
- **Mesh repair** applied for vertebrae (fill_holes + fix_normals + degenerate face removal)

### Step 3 — Fracture Simulation (3 Phases)

Three fracture methods of increasing sophistication, sharing the same post-processing: per-fragment random rotation (±30°), adaptive translation, and fragment count clamped to 2–4.

#### Phase 1 — Random Plane Cutting

Generates 1–3 random cutting planes within the bone bounding box. Uses `trimesh.intersections.slice_mesh_plane` with capped cross-sections. Always cuts the largest remaining fragment to prevent exponential fragment growth.

- **Result**: 120/120, ~36.8s, avg max translation 23.6mm

#### Phase 2 — Voronoi Fracture

Computes principal axes via SVD, places Voronoi seeds along the principal axis with adaptive perpendicular jitter (adjusted for flat bones like vertebrae). Assigns mesh faces to nearest seed via cKDTree, merges fragments <5% volume into the nearest large fragment. Applies random normal-direction perturbation (uniform 0–0.3mm) to fracture surfaces.

- **Result**: 120/120, ~50.3s, avg fragments 2.92, avg boundary ratio 7.1%

#### Phase 3 — Spectral Laplace-Beltrami (LB) Analysis

For 24 samples (2 per bone group), constructs the Laplace-Beltrami operator L and mass matrix M via `robust-laplacian`, solves the generalized eigenvalue problem `L u = λ M u` (scipy sparse eigsh, shift-invert mode). The crack direction is determined by the zero-crossing plane of the first non-trivial eigenvector. Enumerates eigenvectors u₁~u₅ with a scoring function (balance 35% + centrality 35% + planarity 30%) to select the optimal crack plane. Fracture surface Gaussian noise std=0.5mm.

- **Result**: 24/24, ~59.3s (avg 2.5s per case), fixed 2 fragments per sample
- **Long bones** (femur, humerus): clear transverse cracks (<45° to principal axis)
- **Flat bones** (scapula, pelvis): oblique cracks, visually distinct from Voronoi

### Step 4 — Fracture Surface Labeling

Binary labeling (fracture surface = 1, intact surface = 0) applied to Phase 2 (Voronoi) fractured point clouds (120 entries). Phase 1 and Phase 3 generate boundary masks internally during fracture simulation. Uses fragment inter-distance in raw millimeter coordinates: for each point, the minimum distance to points in other fragments is computed via cKDTree. Points with distance < τ = 4.0mm are labeled as fracture surface.

- **Result**: 120/120, ~23.5s, avg F1 = 0.990, avg Precision = 0.993, avg Recall = 0.988
- **F1 ≥ 0.85**: 120/120 (100%), F1 ≥ 0.90: 120/120 (100%)

## Data Source

Uses the **TotalSegmentator v2** small subset (102 cases, 3.2 GB, NIfTI format). Each case contains:
- `ct.nii.gz` — CT volume (Hounsfield Unit voxel matrix)
- `segmentations/<bone_name>.nii.gz` — binary mask for each bone

Available bone labels (total task): femur, humerus, vertebrae (L1–L5, T1–T12, C1–C7), ribs (left/right 1–12), pelvis (hip, sacrum), skull, clavicula, scapula, sternum.

Note: tibia belongs to the `appendicular_bones` subtask in v2 and is **not** present in the small subset.

## Directory Structure

```
data/
  raw/                  # Original NIfTI data
  meshes/               # Step 1 output: s0001_femur_left.ply
  pointclouds/          # Step 2 output: s0001_femur_left.npz
  fractured/            # Step 3 output: s0001_femur_left_frac.npz
  labeled/              # Step 4 output: s0001_femur_left_labeled.npz
results/
  step0_dataset_overview/
  step1_reconstruction/
  step2_pointcloud/
  step3_fracture/       # Subdirs: phase1/, phase2/, phase2_interactive/, phase3/
  step4_label/
code/
  step0_select_cases.py
  step1_reconstruct.py
  step2_sample_pointcloud.py
  step3_fracture_phase1.py
  step3_fracture_phase2.py
  step3_fracture_phase3.py
  step5_pipeline_overview.py
instruction/            # per-step documentation (in Chinese)
```

## Dependencies

- `nibabel` — NIfTI I/O
- `scikit-image` — Marching Cubes surface reconstruction
- `trimesh` — mesh processing, sampling, plane cutting
- `numpy`, `scipy` — data processing, cKDTree, sparse eigenvalue solver
- `matplotlib` — all visualizations
- `robust-laplacian` — Laplace-Beltrami operator construction (Phase 3)
- `plotly` — interactive 3D HTML visualization (Phase 2)

## Environment

- Windows 11, CPU only
- Single case processing < 15 minutes
- All intermediate results persisted to disk with resume support
- Point cloud size: N = 4096 (fixed)

## Comparison with the Original Paper

| Aspect | Original (Yuan Qinhui) | This Pipeline |
|--------|----------------------|---------------|
| Data source | Hospital CT DICOM → Mimics | TotalSegmentator NIfTI → Marching Cubes |
| Bone type | Tibial plateau only | 13 bone groups (femur, vertebrae, ribs, etc.) |
| Fracture method | FEM modal analysis (Sellan 2023) with material parameters | Plane cutting → Voronoi → Spectral LB modal analysis (geometry-based, no material parameters) |
| Surface labeling | Geometric features + manual validation (97% agreement) | Fragment inter-distance method (F1=0.990) |
| Deep learning | PointNet++ training included | Not included (data prep only) |

## Key Results Summary

| Step | Metric | Result |
|------|--------|--------|
| Step 1 Reconstruction | Success rate | 120/120 (100%) |
| Step 2 Sampling | Success rate | 120/120 (100%) |
| Step 3 Phase 1 | Success rate | 120/120 (100%) |
| Step 3 Phase 2 | Success rate, avg boundary ratio | 120/120 (100%), 7.1% |
| Step 3 Phase 3 | Success rate, avg boundary ratio | 24/24 (100%), avg boundary 12.7%, long bones crack angle <45° |
| Step 4 Labeling | Avg F1, Avg Precision, Avg Recall | 0.990, 0.993, 0.988 |

## Usage

Activate the `pointnet` conda environment and run scripts sequentially:

```powershell
conda activate pointnet
python code/step0_select_cases.py
python code/step1_reconstruct.py
python code/step2_sample_pointcloud.py
python code/step3_fracture_phase1.py
python code/step3_fracture_phase2.py
python code/step3_fracture_phase3.py
python code/step4_label.py
python code/step5_pipeline_overview.py
```

Each script supports resume: already-processed entries are automatically skipped.

## License

This project is for research and educational purposes. The TotalSegmentator data is subject to its original license terms.

---

State Key Laboratory of Digital Medical Engineering
School of Biological Science and Medical Engineering, Southeast University; Contact: 213230182@seu.removethis.edu.cn
