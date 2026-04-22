# Nasal Small-Structure Data Generator

**Author:** Dilip Goswami, Berlin, Germany

A high-quality synthetic data generation pipeline for **nasal endoscopy image segmentation**, designed to improve model performance on **small and challenging anatomical structures**.

This project focuses on **quality over quantity** by combining strict crop selection, object-preserving constraints, realistic endoscopic augmentation, and duplicate rejection to produce a balanced segmentation dataset.

---

## Overview

Training segmentation models on small nasal structures is difficult because of:

* class imbalance,
* limited visibility,
* blurry frames,
* inconsistent annotations,
* and low-value crops that confuse the model.

This generator addresses those issues by building a curated synthetic dataset from an existing annotated source dataset.

### Main goals

* Generate approximately **1,300 high-quality samples**
* Prioritize **rare and small structures**
* Preserve **near-complete object visibility**
* Reject **blurry and duplicate samples**
* Simulate **realistic endoscopic imaging conditions**

---

## What the script does

The script:

1. Scans a Supervisely-style dataset structure containing images and annotations
2. Parses annotation geometries including polygons, rectangles, and bitmaps
3. Normalizes class labels using aliases and fuzzy matching
4. Builds class-specific crop candidates around annotated structures
5. Enforces strict object containment and foreground-density checks
6. Applies realistic endoscopic augmentation
7. Resizes outputs to a standard resolution
8. Saves paired image and annotation files in a structured output layout
9. Uses image-mask hashing to reduce duplicate samples
10. Writes dataset summary statistics after generation

---

## Target classes

The current generation plan focuses on the following anatomical structures:

* Arc choanal
* Opercule du cornet moyen
* Cornet inférieur
* Cornet moyen
* Septum

These classes are intentionally weighted to improve representation of **small, rare, or difficult structures**.

---

## Key features

### 1. Class-aware sampling

Rare and difficult classes receive more generated samples, while abundant classes are limited to reduce dataset bias.

### 2. Strict crop quality control

Generated crops are filtered to ensure:

* high sharpness,
* sufficient foreground presence,
* and near-complete target-object visibility.

### 3. Endoscopic realism

The augmentation pipeline simulates common endoscopic image characteristics such as:

* subtle sensor noise,
* brightness drift,
* and local contrast variation.

### 4. Duplicate rejection

A combined image-and-mask hashing strategy helps prevent repeated low-diversity samples.

### 5. Left/right anatomical separation

Output samples are organized by inferred nasal side for structured downstream usage.

---

## Dataset assumptions

This script expects a dataset layout similar to:

```text
Training/
├── Patient_01/
│   ├── img/
│   │   ├── 0001.jpg
│   │   └── 0002.jpg
│   └── ann/
│       ├── 0001.json
│       └── 0002.json
├── Patient_02/
│   ├── img/
│   └── ann/
```

Each dataset leaf should contain:

* an `img/` folder with source images,
* an `ann/` folder with corresponding annotation JSON files.

---

## Output structure

The generated dataset is saved in a structure like:

```text
SYNTH_GoldStandard/
└── P100_blockframes_subselection_supervisely/
    ├── LEFTNOSE_B1/
    │   ├── img/
    │   ├── ann/
    │   └── img_info/
    ├── RIGHTNOSE_B1/
    │   ├── img/
    │   ├── ann/
    │   └── img_info/
    ├── registry.json
    └── dataset_summary.json
```

---

## Installation

### Requirements

* Python 3.9+
* NumPy
* Pillow

### Install dependencies

```bash
pip install numpy pillow
```

---

## Usage

```bash
python nasal_gold_standard_data_generator.py --src <input_dataset_path> --dst <output_dataset_path>
```

### Example

```bash
python nasal_gold_standard_data_generator.py \
  --src "/path/to/Training" \
  --dst "/path/to/SYNTH_GoldStandard"
```

### Optional argument

Disable fuzzy label normalization:

```bash
python nasal_gold_standard_data_generator.py --src <input> --dst <output> --no-fuzzy
```

---

## Core generation strategy

The pipeline is designed around a **quality-first philosophy**:

* **Object-centered crops** are generated around target annotations
* **Strict containment checks** reject crops that truncate the target anatomy
* **Foreground density thresholds** reject weak or low-information crops
* **Blur filtering** removes visually poor samples
* **Augmentation** increases realism without destroying anatomical consistency
* **Deduplication** avoids overfitting from repeated examples

This makes the dataset especially useful for segmentation models that struggle on **small boundaries and fine anatomical detail**.

---

## Configuration highlights

The script includes configurable parameters for:

* target sample counts per class,
* crop scale ranges,
* minimum crop size,
* blur threshold,
* augmentation frequency,
* and foreground acceptance thresholds.

These settings can be adjusted depending on:

* dataset quality,
* model sensitivity,
* class imbalance,
* and desired output size.

---

## Limitations

This script does **not**:

* train a segmentation model,
* evaluate model performance,
* or generate annotations from unlabeled images.

It is a **data preparation and dataset enrichment tool** for supervised segmentation workflows.

---

## Recommended use cases

This project is useful for:

* medical image segmentation research,
* endoscopic vision datasets,
* rare-class balancing,
* boundary-sensitive training,
* and synthetic dataset expansion for small anatomical targets.

---

## Citation

If you use this work in a research or development setting, please cite or acknowledge:

**Dilip Goswami, Berlin, Germany**

---

## License

Add your preferred license here, for example:

* MIT License
* Apache-2.0
* BSD-3-Clause

---

## Contact

For research collaboration, adaptation, or improvement of the pipeline, please open an issue or add your contact details here.
