# Exoplanet Machine Learning Project

This project builds machine learning models to predict exoplanet type (e.g., Terran, Jovian) and planet mass using a real-world dataset of observed planetary characteristics. The workflow covers data preprocessing, exploratory analysis, model training, hyperparameter tuning with cross-validation, and evaluation.

Multiple models were tested, including logistic/linear regression, decision trees, and random forests. Results show that tree-based models outperform linear models for regression, likely due to nonlinear relationships and a highly skewed target distribution, while classification performance benefits from both linear and nonlinear approaches. Feature importance analysis was used to interpret model behavior and identify key predictive variables.

The dataset contains 40+ features and is too large to preview directly on GitHub, but it is included in the repository for download. A preview and summary statistics are available in the notebook.

Dataset: [Exoplanet Dataset (Kaggle)](https://www.kaggle.com/datasets/chandrimad31/phl-exoplanet-catalog?resource=download)

## Structure
notebooks/analysis.ipynb   # Main analysis and results

src/preprocessing.py       # Data cleaning & feature engineering

src/modeling.py            # Model training, tuning, evaluation

data/                      # Dataset (download required)


## Tech Stack
- Python (pandas, NumPy, scikit-learn)
- matplotlib / seaborn
