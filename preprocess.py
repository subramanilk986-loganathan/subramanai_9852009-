import pandas as pd

def load_data(path):
    """Load CSV and basic cleaning."""
    df = pd.read_csv(path)
    # Basic cleaning example: drop fully empty rows and reset index
    df = df.dropna(how='all').reset_index(drop=True)
    return df

def basic_preprocess(df):
    """Return X, y after simple preprocessing.
    Assumes target column is named 'target' or 'num' (common variants).
    """
    df = df.copy()
    # Try common column names for target
    if 'target' in df.columns:
        target_col = 'target'
    elif 'num' in df.columns:
        target_col = 'num'
    else:
        # if not found, assume last column is target
        target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
