import numpy as np
import pandas as pd


def load_data(train_path: str, test_path: str, submission_path: str):
    train = pd.read_csv(train_path, parse_dates=["Datetime"])
    test  = pd.read_csv(test_path,  parse_dates=["Datetime"])
    ss    = pd.read_csv(submission_path)
    print(f"[OK] Train shape: {train.shape}")
    print(f"[OK] Test shape:  {test.shape}")
    return train, test, ss


def impute_missing(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """
    Impute missing values using train medians.
    Train medians are applied to test to avoid data leakage.
    """
    nan_cols = [
        "Sensor1_PM2.5", "Sensor2_PM2.5",
        "Temperature", "Relative_Humidity"
    ]
    for col in nan_cols:
        median_val  = train[col].median()
        train[col]  = train[col].fillna(median_val)
        test[col]   = test[col].fillna(median_val)

    print(f"[OK] Missing values imputed using train medians")
    return train, test


def log_transform(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """
    Log-transform skewed sensor columns to compress right tails
    and improve gradient boosting split quality.
    """
    log_cols = [
        "Relative_Humidity", "Temperature",
        "Sensor1_PM2.5", "Sensor2_PM2.5"
    ]
    for dataset in (train, test):
        dataset[log_cols] = np.log(dataset[log_cols] + 1)

    print(f"[OK] Log transformation applied to: {log_cols}")
    return train, test