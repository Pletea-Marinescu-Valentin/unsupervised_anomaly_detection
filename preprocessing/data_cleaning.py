import pandas as pd

def remove_missing_values(df):
    # Drop rows with missing values
    df = df.dropna()
    print(f"Removed missing values. Remaining rows: {len(df)}")
    return df

def remove_outliers(df, columns):
    """Remove outliers using Z-score."""
    for col in columns:
        initial_count = len(df)
        df = df[(df[col] - df[col].mean()).abs() / df[col].std() <= 3]
        print(f"Removed outliers in {col}. Rows reduced from {initial_count} to {len(df)}.")
    return df
