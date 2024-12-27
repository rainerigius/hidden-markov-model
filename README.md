# Hidden Markov Model (HMM) Trading Project
https://colab.research.google.com/drive/1r7XeSxH5v--EfhCDpjZmJ_mIWz9IfoyM#scrollTo=1p-u99S5ZxEf
## Overview

This repository contains implementations of several Hidden Markov Models (HMM) designed to analyze trading data with various levels of indicator integration and correction methods. The models achieve different performance accuracies, with some versions reaching up to **97% accuracy** based on backtesting metrics.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/rainerigius/hidden-markov-model.git
    ```

2. **Install dependencies:** To ensure proper functionality, install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

### Essential Packages

- **hmmlearn:** For training and evaluating Hidden Markov Models.
- **numpy, pandas:** For data manipulation and numerical operations.
- **joblib:** For saving and loading model files.
- **scikit-learn:** For data preprocessing, scaling, and other utility functions.
- **matplotlib, seaborn:** For data visualization.

## Files and Scripts

### Main HMM Scripts

The repository includes several Python scripts, each implementing a different HMM with varying configurations of indicators and metrics:

- `hmm_87%30_ind+_correction.py`: Implements an HMM model with 87% accuracy, utilizing 30 indicators and a correction method.
- `hmm_87%_with_30_indicators.py`: HMM model achieving 87% accuracy with 30 indicators.
- `hmm_88%.py`: An HMM model with a slightly higher accuracy of 88%.
- `hmm_97%_updated_metrics.py`: A refined version of the 97% accuracy model, with updated metrics and performance improvements.
- `hmm_97%.py`: A previous version of the 97% accuracy model.
- `hmm_d_97%.py`: Another version of the 97% accuracy model, potentially using different datasets or indicators.

### Old Template

- `oldtemplate.py`: The original template for implementing HMM models, which provides a foundational structure for building more advanced models.

### Additional Tools and Data Files

- `liquidity.py`: Script that calculates liquidity metrics for trading data.
- `oos_test.py`: Out-of-sample testing script for evaluating model performance on unseen data.
- `state_transition_diagram`: Visualization of the state transitions for the HMM models.

### Datasets

The repository includes a few CSV files that contain sample data:

- `btc.csv`, `BTC_1H.csv`, `csv/BTC_2H.csv`, etc.: Bitcoin price data at various timeframes.
- `data/bitcoin_state_changes.csv`: Data capturing state transitions for Bitcoin, likely used in the HMM training process.

### Model Files

Pre-trained models are saved in the `models` directory with joblib:

- `model_hmm_85%_30ind_updated.joblib`: A pre-trained HMM model with 85% accuracy using 30 indicators.
- `model_hmm_88%.joblib`: A pre-trained HMM model with 88% accuracy.
- `model_hmm_98%.joblib`: A highly accurate pre-trained HMM model with 98% accuracy.

## How to Use the HMM Models

Each model script can be executed directly or used as part of a larger analysis pipeline. For example, to run the `hmm_97%_updated_metrics.py` model, execute:

```bash
python hmm_97%_updated_metrics.py
```
Results are printed to the console or saved in designated output files for review.

## Notes on HMMs and Project Structure
Hidden Markov Models (HMMs) are statistical models that assume the system being modeled is a Markov process with hidden states. In this project, the HMMs are trained on historical trading data, aiming to predict price movements based on various indicators. Each HMM script uses different sets of indicators and configurations to optimize performance. Accuracy percentages indicate the effectiveness of each model based on backtesting metrics.
