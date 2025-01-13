from sklearn.preprocessing import StandardScaler

def scale_data(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    print("Data scaled successfully.")
    return df

