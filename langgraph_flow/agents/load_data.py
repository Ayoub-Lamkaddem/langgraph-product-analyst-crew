import pandas as pd
from .state import DataState

def load_data(state: DataState) -> DataState:
    df = state["df"].copy()

    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    df = df.drop_duplicates()
    threshold = len(df) * 0.5
    df = df.dropna(axis=1, thresh=threshold)

    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object','category']).columns

    for col in numeric_cols:
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)
    
    for col in categorical_cols:
        mode_value = df[col].mode()
        df[col].fillna(mode_value[0], inplace=True)

    for col in categorical_cols:
        df[col] = df[col].str.strip().str.lower()

    state["processed_df"] = df
    return state