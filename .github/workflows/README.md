# GitHub Actions MLOps Lab 2 - Iris Classification

This project is my customized version of GitHub Lab 2 from the MLOps lab collection.

## What I changed
- Used the Iris dataset instead of synthetic classification data
- Trained a RandomForest classifier with custom parameters
- Saved versioned model artifacts in the `models/` folder
- Saved evaluation metrics in the `metrics/` folder
- Added a calibration workflow
- Added simple tests and an improved README

## Workflow
1. Push code to `main`
2. GitHub Actions installs dependencies
3. Model is trained automatically
4. Metrics are generated automatically
5. Artifacts are committed back to the repository
6. Calibration workflow creates a calibrated model version

## Output
- Trained models in `models/`
- Evaluation reports in `metrics/`

## Submission note
This repo is based on the lab concept, but I modified the dataset, scripts, workflow structure, and documentation.