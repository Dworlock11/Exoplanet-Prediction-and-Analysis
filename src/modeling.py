import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, root_mean_squared_error
from sklearn.inspection import permutation_importance
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning


def data_spliting(df, stratify, target):
    """
    Splits the data into training and testing sets, and identifies numerical and categorical features.
    """

    X = df.drop(target, axis=1)
    y = df[target]

    if stratify == True: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=9)        
    else: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

    num_features = X_train.select_dtypes(include=np.number)
    cat_features = X_train.select_dtypes(exclude=np.number)
    num_col_names = num_features.columns
    cat_col_names = cat_features.columns

    return X, y, X_train, X_test, y_train, y_test, num_features, cat_features, num_col_names, cat_col_names


def optimizing_training_predicting(pipe, param_dist, kf, scorer, X_train, y_train, X_test):
    """
    Cross-validates the model, tunes hyperparameters, predicts the target, and returns the best model and the predicted values.
    """

    simplefilter("ignore", category=ConvergenceWarning)

    if scorer is not None: search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=10, cv=kf, scoring=scorer, 
                                                       random_state=9, n_jobs=-1)
    else: search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=10, cv=kf, random_state=9, n_jobs=-1)            
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    print("Best Parameters:\n")
    for param, value in search.best_params_.items(): print(param.split("__", 1)[1], ":" , value)
    mean_score = search.cv_results_['mean_test_score'][search.best_index_]
    std_score = search.cv_results_['std_test_score'][search.best_index_]
    print(f"\nBest CV Score: {mean_score:.4f} ± {std_score:.4f}")

    y_pred = best_model.predict(X_test) # type: ignore

    return best_model, y_pred


def classification_evaluation(y_test, y_pred, model_name):
    """
    Evaluates the model using a classification report and a confusion matrix.
    """

    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

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


def regression_scoring(y_test, y_pred):
    """
    Evaluates the regression model using mean absolute error and root mean squared error, and prints the factors by which the 
    predictions deviate from the actual values. If the metrics are too large to be calculated, it catches the overflow error and 
    prints a message instead.
    """

    try:
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        print(f"MAE factor  : {10**mae:.2f}×")
        print(f"RMSE factor : {10**rmse:.2f}×")
    except OverflowError as e:
        print("Overflow Error. The metrics were too large to be calculated.")


def regression_plots(y_test, y_pred):
    """
    Plots the actual vs the predicted values for the regression model, using log scale for better visualization 
    of the mass distribution.
    """

    plt.scatter(x=range(0, len(y_test)), y=10**(y_test.reset_index(drop=True)), color="blue", alpha=0.4, label="Actual")
    plt.scatter(x=range(0, len(y_pred)), y=10**y_pred, color="green", alpha=0.4, label="Predicted")
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Planet")
    plt.ylabel("Mass")
    plt.title("Actual vs Predicted Mass")
    plt.show()


def feature_importance(best_model, model_preprocessor, model, X_test, y_test, scorer, model_name):
    """
    Calculates and plots feature importance using permutation importance, showing 
    the top ten most important features in a bar chart.
    """

    preprocessor = best_model.named_steps[model_preprocessor]
    model_steps = best_model.named_steps[model]

    raw_feature_names = preprocessor.get_feature_names_out()

    clean_feature_names = [
        name.split("__", 1)[1]
        if "__" in name else name
        for name in raw_feature_names
    ]

    X_test_transformed = preprocessor.transform(X_test)

    if scorer is not None: importances = permutation_importance(model_steps, X_test_transformed, y_test, scoring=scorer, n_repeats=10, random_state=9, n_jobs=-1)
    else: importances = permutation_importance(model_steps, X_test_transformed, y_test, n_repeats=10, random_state=9, n_jobs=-1)

    highest_importances = pd.Series(importances.importances_mean, index=clean_feature_names).sort_values(ascending=False).head(10) # type: ignore

    plt.bar(highest_importances.index, highest_importances)
    plt.xticks(rotation=70)
    plt.title(f"{model_name} Feature Importance")
    plt.xlabel("Feature")
    plt.ylabel("Drop in Performance")
    plt.show()
