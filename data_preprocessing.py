import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, root_mean_squared_error
from sklearn.inspection import permutation_importance
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

# Column removal based on null values and feature selection
def remove_cols(df):
    # Remove null columns
    col_non_null_count = df.isna().sum()
    cols_non_majority_null = col_non_null_count[col_non_null_count < len(df)/4].index.to_list()
    df = df[cols_non_majority_null]

    # Additional feature selection
    df = df.drop(["P_NAME", "P_STATUS", "P_RADIUS", "P_YEAR", "P_UPDATED", "S_NAME", "S_RADIUS", "S_ALT_NAMES", "P_HABZONE_OPT", "P_HABZONE_CON", "S_CONSTELLATION_ABR", "P_PERIOD_ERROR_MIN", "P_PERIOD_ERROR_MAX", "S_DISTANCE_ERROR_MIN", "S_DISTANCE_ERROR_MAX", "P_FLUX_MIN", "P_FLUX_MAX", "P_TEMP_EQUIL_MIN", "P_TEMP_EQUIL_MAX"], axis=1)

    # Find categorical features
    cat_features = df.select_dtypes(exclude=np.number)

    # Number of null values per feature
    print("Value count per feature:")
    for col in cat_features.columns:
        print(col, "-", len(cat_features[col].value_counts()))

    # Drop features with too many different values
    df = df.drop(["S_RA_T", "S_DEC_T", "S_CONSTELLATION", "S_CONSTELLATION_ENG"], axis=1)

    return df


# Mask Miniterran as Subterran and print mass information
def mini_terran(df):
    # Miniterran mass
    miniterran = df[df["P_TYPE"] == "Miniterran"]
    print("Miniterran mass:", f"{miniterran["P_RADIUS_EST"].iloc[0]:.3f}")

    # Subterran minimum mass
    subterran = df[df["P_TYPE"] == "Subterran"]
    print("Smallest Subterran mass:", f"{subterran["P_RADIUS_EST"].min():.3f}")

    # Mask as Subterran
    df["P_TYPE"] = df["P_TYPE"].mask(df["P_TYPE"] == "Miniterran", "Subterran")

    return df


# Plot mass distribution of exoplanets
def mass_dist(df):
    plt.scatter(x=range(0, len(df.index)), y=df["P_MASS_EST"].sort_values(ascending=False))
    plt.xlabel("Planet")
    plt.ylabel("Mass")
    plt.title("Sorted Mass of Exoplanets")
    plt.show()


# Define Huber loss function for regression evaluation
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    abs_error = np.abs(error)

    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic

    return np.mean(0.5 * quadratic**2 + delta * linear)


# Apply log transformation to mass and drop original mass column
def log_space(df):
    log_df = df.copy()
    log_df["Log_Mass"] = np.log10(log_df["P_MASS_EST"])
    log_df = log_df.drop("P_MASS_EST", axis=1)

    return log_df


# Remove entries with null values in the target column
def remove_null_entries(df, col):
    print(f"Number of null values in {col}: {df[col].isna().sum()}")
    target_na = df[df["P_TYPE"].isna()].index
    df = df.drop(target_na)
    print(f"Number of null values after removal: {df[col].isna().sum()}")

    return df


# Split data into training and testing sets, and identify numerical and categorical features
def data_spliting(df, stratify, target):
    # Define features and target
    X = df.drop(target, axis=1)
    y = df[target]

    # Train-test split
    if stratify == True: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=9)
    else: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

    # Find numerical and categorical columns
    num_features = X_train.select_dtypes(include=np.number)
    cat_features = X_train.select_dtypes(exclude=np.number)
    num_col_names = num_features.columns
    cat_col_names = cat_features.columns

    return X, y, X_train, X_test, y_train, y_test, num_features, cat_features, num_col_names, cat_col_names


# Optimize model training and prediction using RandomizedSearchCV, and return the best model and predictions
def optimizing_training_predicting(pipe, param_dist, kf, scorer, X_train, y_train, X_test):
    # Supress convergence warnings
    simplefilter("ignore", category=ConvergenceWarning)

    # Search for best model, extract it, and print optimized parameters
    if scorer is not None: search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=10, cv=kf, scoring=scorer, random_state=9, n_jobs=-1)
    else: search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=10, cv=kf, random_state=9, n_jobs=-1)    
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print("Best Parameters:\n")
    for param, value in search.best_params_.items(): print(param.split("__", 1)[1], ":" , value)

    # Predict the target
    y_pred = best_model.predict(X_test) # type: ignore

    return best_model, y_pred


# Evaluate classification model using classification report and confusion matrix
def classification_evaluation(y_test, y_pred, model_name):
    # Print classification report
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    print("Confusion Matrix:\n")
    class_labels = ["Jovian", "Neptunian", "Subterran", "Superterran", "Terran"]
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues) # type: ignore
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xticks(rotation=45)
    plt.show()


# Evaluate regression model using MAE and RMSE, and print the factors by which predictions deviate from actual values
def regression_scoring(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"MAE factor  : {10**mae:.2f}×")
    print(f"RMSE factor : {10**rmse:.2f}×")


# Plot actual vs predicted values for regression model, using log scale for better visualization of mass distribution
def regression_plots(y_test, y_pred):
    plt.scatter(x=range(0, len(y_test)), y=10**(y_test.reset_index(drop=True)), color="blue", alpha=0.4, label="Actual")
    plt.scatter(x=range(0, len(y_pred)), y=10**y_pred, color="green", alpha=0.4, label="Predicted")
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Planet")
    plt.ylabel("Mass")
    plt.title("Actual vs Predicted Mass")
    plt.show()


# Calculate and plot feature importance using permutation importance, showing the top 10 most important features for the model
def feature_importance(best_model, model_preprocessor, model, X_test, y_test, model_name):
    # Extract components
    preprocessor = best_model.named_steps[model_preprocessor]
    model_steps = best_model.named_steps[model]

    raw_feature_names = preprocessor.get_feature_names_out()

    # Remove transformer names from features
    clean_feature_names = [
        name.split("__", 1)[1]
        if "__" in name else name
        for name in raw_feature_names
    ]

    # Transform X_test into expanded feature space
    X_test_transformed = preprocessor.transform(X_test)

    # Run permutation importance
    importances = permutation_importance(model_steps, X_test_transformed, y_test, n_repeats=10, random_state=9, n_jobs=-1)

    # Display results
    highest_importances = pd.Series(importances.importances_mean, index=clean_feature_names).sort_values(ascending=False).head(10) # type: ignore

    plt.bar(highest_importances.index, highest_importances)
    plt.xticks(rotation=70)
    plt.title(f"{model_name} Feature Importance")
    plt.xlabel("Feature")
    plt.ylabel("Drop in Performance")
    plt.show()