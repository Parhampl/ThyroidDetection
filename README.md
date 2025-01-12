
# Thyroid Detection

This repository contains a machine learning pipeline for predicting thyroid health status using classical machine learning methods.

## Problem Statement
The objective is to predict the thyroid status (`Thyroid Status` column) based on the other features in the dataset. This project implements the following:
- Preprocessing: Encoding categorical features and splitting data into training/testing sets.
- Model: A Random Forest Classifier to predict the thyroid status.
- Evaluation: Calculates accuracy, precision, recall, and F1-score for each class.

## Files and Folders
- `data/`: Contains the dataset `ThyroidDetection.csv`.
- `notebooks/`: Jupyter Notebook with the full pipeline (`ML_First_Practice.ipynb`).
- `scripts/`: Python script (`thyroid_detection.py`) for running the analysis.
- `docs/`: Includes the problem statement in Persian (`تمرین یادگیری ماشین.pdf`).

## How to Run
### Prerequisites
- Python 3.8 or later
- Required libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ThyroidDetection.git
   cd ThyroidDetection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python scripts/thyroid_detection.py
   ```

## Results
The model provides evaluation metrics such as:
- Overall accuracy: ~XX% (based on the dataset)
- Class-wise metrics: Precision, Recall, and F1-score for each class.

## Dataset
The dataset (`ThyroidDetection.csv`) includes features relevant to thyroid health diagnosis.

## Acknowledgments
- Problem statement from the file `تمرین یادگیری ماشین.pdf`.
- Machine learning implementation using scikit-learn.
