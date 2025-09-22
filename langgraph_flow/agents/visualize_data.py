import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from .state import DataState

def visualize(state: DataState) -> dict:
    df = state['processed_df']
    figs = {"hist": [], "heatmap": None}
    
    # Colonnes numériques pour histogrammes
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        fig = px.histogram(df, x=col, title=f"Distribution de {col}")
        figs["hist"].append(fig)

    # Heatmap des corrélations
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12,8))
        heatmap_fig = sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
        figs["heatmap"] = plt.gcf()
        plt.close()

    return figs
