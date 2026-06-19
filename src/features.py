import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from raw sensor readings.

    Features created:
    - Datetime decomposition: hour, day, month, year, day-of-week,
      week-of-year to capture temporal sensor behaviour patterns
    - PM2.5 bins: discretized ranges to capture threshold effects
    - S1_S2 delta: divergence between sensors — a strong fault signal
    - IsWeekend: weekend flag based on maintenance cycle patterns
    - IsHot: above-average temperature flag for stress detection
    """
    df = df.copy()

    # Datetime decomposition
    df["Datetime_day"]          = df["Datetime"].dt.day
    df["Datetime_month"]        = df["Datetime"].dt.month
    df["Datetime_year"]         = df["Datetime"].dt.year
    df["Datetime_hour"]         = df["Datetime"].dt.hour
    df["Datetime_minute"]       = df["Datetime"].dt.minute
    df["Datetime_seconds"]      = df["Datetime"].dt.second
    df["Datetime_day_of_year"]  = df["Datetime"].dt.dayofyear
    df["Datetime_day_of_week"]  = df["Datetime"].dt.dayofweek
    df["Datetime_week_of_year"] = df["Datetime"].dt.isocalendar().week.astype(int)
    df.drop("Datetime", axis=1, inplace=True)

    # PM2.5 bin features
    bins   = [-np.inf, 3.0, 3.5, 4.0, np.inf]
    labels = [1, 2, 3, 4]
    df["Sensor1_bins"] = pd.cut(df["Sensor1_PM2.5"], bins=bins, labels=labels)
    df["Sensor2_bins"] = pd.cut(df["Sensor2_PM2.5"], bins=bins, labels=labels)

    # Inter-sensor delta — key fault signal
    df["S1_S2"] = df["Sensor1_PM2.5"] - df["Sensor2_PM2.5"]

    # Behavioural flags
    df["IsWeekend"] = (df["Datetime_day_of_week"] >= 5).astype(int)
    df["IsHot"]     = (df["Temperature"] > df["Temperature"].mean()).astype(int)

    # Drop ID if present
    if "ID" in df.columns:
        df.drop("ID", axis=1, inplace=True)

    df = df.dropna()
    return df