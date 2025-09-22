import pandas as pd
from typing import TypedDict

class DataState(TypedDict):
    df: pd.DataFrame
    processed_df: pd.DataFrame