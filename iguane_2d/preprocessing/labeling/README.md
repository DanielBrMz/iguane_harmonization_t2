# View Labeling Scripts

This directory contains tools for labeling fetal brain MRI views (sagittal, axial, coronal).

## Scripts

### `auto_classify_views.py`
Semi-automated view classification using feature extraction and clustering.
- Extracts discriminative features (left-right symmetry, intensity patterns)
- Uses k-means clustering for initial classification
- Provides manual correction interface

### `batch_label_views.py`
Non-interactive batch labeling workflow for large datasets.
- Generates image sheets (PNG files) for manual review
- Creates CSV templates for label entry
- Reads labels from CSV after manual annotation
- Ideal for labeling many images offline

### `manual_label_views.py`
Interactive real-time labeling interface with keyboard controls.
- Simple keyboard-based navigation
  - `s` - Sagittal view
  - `a` - Axial/Coronal view
  - `u` - Unknown/Unclear
  - `b` - Go back
  - `q` - Quit and save
- Fast and efficient for moderate-sized datasets
- Shows images one at a time with instant feedback

## Usage

Choose the appropriate tool based on your needs:
- **Auto-classify** first for initial labels, then manual correction
- **Batch labeling** for very large datasets or team labeling
- **Manual labeling** for quick interactive labeling sessions
