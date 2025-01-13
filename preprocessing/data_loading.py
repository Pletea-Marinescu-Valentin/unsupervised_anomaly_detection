import pandas as pd

def load_dataset(file_path):
    import pandas as pd

    # Load the dataset
    df = pd.read_csv(file_path)

    # Check the first few rows
    print(df.head())

    return df

