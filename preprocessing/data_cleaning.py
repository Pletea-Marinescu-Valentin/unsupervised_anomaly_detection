import pandas as pd

def remove_missing_values(df):
    # Drop rows with missing values
    df = df.dropna()
    print(f"Removed missing values. Remaining rows: {len(df)}")
    return df

def remove_outliers(df, columns):
    """Remove outliers using Z-score."""
    for col in columns:
        df = df[(df[col] - df[col].mean()).abs() / df[col].std() <= 3]
    return df
