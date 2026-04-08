import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def remove_cols(df):
    """
    Removes the columns with too many null values, those which serve as bad predictors, and those with too many different values.
    """
    
    col_non_null_count = df.isna().sum()
    cols_non_majority_null = col_non_null_count[col_non_null_count < len(df)/4].index.to_list()
    df = df[cols_non_majority_null]

    df = df.drop(["P_NAME", "P_STATUS", "P_RADIUS", "P_YEAR", "P_UPDATED", "S_NAME", "S_RADIUS", "S_ALT_NAMES", "P_HABZONE_OPT", 
                  "P_HABZONE_CON", "S_CONSTELLATION_ABR", "P_PERIOD_ERROR_MIN", "P_PERIOD_ERROR_MAX", "S_DISTANCE_ERROR_MIN", 
                  "S_DISTANCE_ERROR_MAX", "P_FLUX_MIN", "P_FLUX_MAX", "P_TEMP_EQUIL_MIN", "P_TEMP_EQUIL_MAX"], axis=1)

    cat_features = df.select_dtypes(exclude=np.number)

    print("Value count per feature:\n")
    for col in cat_features.columns:
        print(col, "-", len(cat_features[col].value_counts()))

    df = df.drop(["S_RA_T", "S_DEC_T", "S_CONSTELLATION", "S_CONSTELLATION_ENG"], axis=1)

    return df


def mini_terran(df):
    """
    Masks Miniterrans as Subterrans.
    """

    miniterran = df[df["P_TYPE"] == "Miniterran"]
    print("Miniterran mass:", f"{miniterran["P_RADIUS_EST"].iloc[0]:.3f}")

    subterran = df[df["P_TYPE"] == "Subterran"]
    print("Smallest Subterran mass:", f"{subterran["P_RADIUS_EST"].min():.3f}")

    df["P_TYPE"] = df["P_TYPE"].mask(df["P_TYPE"] == "Miniterran", "Subterran")

    return df


def mass_dist(df):
    """
    Plots the distribution of exoplanet masses, sorted in descending order.
    """

    plt.scatter(x=range(0, len(df.index)), y=df["P_MASS_EST"].sort_values(ascending=False))
    plt.xlabel("Planet")
    plt.ylabel("Mass")
    plt.title("Sorted Mass of Exoplanets")
    plt.show()


def radius_dist(df):
    """
    Plots the distribution of exoplanet radii, sorted in descending order.
    """
    plt.scatter(x=range(0, len(df.index)), y=df["P_RADIUS_EST"].sort_values(ascending=False))
    plt.xlabel("Planet")
    plt.ylabel("Radius")
    plt.title("Sorted Radius of Exoplanets")
    plt.show()


def huber_loss(y_true, y_pred, delta=1.0):
    """
    Define the Huber loss function for regression evaluation.
    """

    error = y_true - y_pred
    abs_error = np.abs(error)

    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic

    return np.mean(0.5 * quadratic**2 + delta * linear)


def log_space(df):
    """
    Applies a log transformation to the mass and radius columns and drops the original columns.
    """

    log_df = df.copy()
    log_df["Log_Mass"] = np.log10(log_df["P_MASS_EST"])
    log_df["Log_Radius"] = np.log10(log_df["P_RADIUS_EST"])
    log_df = log_df.drop(["P_MASS_EST", "P_RADIUS_EST"], axis=1)

    return log_df


def remove_null_entries(df, col):
    """
    Removes the entries with null values in the target column, and prints the number of null values before and after removal.
    """

    print(f"Number of null values in {col}: {df[col].isna().sum()}")
    target_na = df[df["P_TYPE"].isna()].index
    df = df.drop(target_na)
    print(f"Number of null values after removal: {df[col].isna().sum()}")

    return df