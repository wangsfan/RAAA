# lifelong_methods

This package provides shared buffer and sampling utilities for the IIRC training pipelines.

## Contents

- **buffer/buffer.py** — Experience replay buffer for the DyTox-IIRC baseline (`main_iirc.py`).
- **buffer/buffer_hcast.py** — Experience replay buffer for the HSFS method (`main_hsfs.py`), adapted to handle superpixel segmentation inputs.
- **utils.py** — General utilities: `SubsetSampler`, optimizer construction, and data transform helpers shared across training pipelines.
