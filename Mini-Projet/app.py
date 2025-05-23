# streamlit : pour créer l'interface web.
import streamlit as st
# pandas : pour manipuler ton fichier CSV.
import pandas as pd
# seaborn et matplotlib : pour faire les graphes.
import seaborn as sns
import matplotlib.pyplot as plt
# PCA, StandardScaler, KMeans : pour réduire les dimensions (PCA), normaliser les données (Scaler) et créer des groupes (clustering).
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# LangGraph : pour organiser les étapes de traitement comme un pipeline intelligent.
from langgraph.graph import StateGraph
# TypedDict : pour définir un état structuré (le format des données partagées entre les étapes).
from typing import TypedDict

# ------------ 1. Shared State ------------
class DataState(TypedDict):
    df: pd.DataFrame
    processed_df: pd.DataFrame

# ------------ 2. Agent Functions ------------

def load_data(state: DataState) -> DataState:
    df = state["df"]

    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors='coerce')
    df.dropna(subset=["Date"], inplace=True)

    state["df"] = df
    return state

def inspect_data(state: DataState) -> DataState:
    df = state["df"]
    st.subheader("Aperçu des données")
    st.write(df.describe())
    st.write("Valeurs manquantes :")
    st.write(df.isnull().sum())
    return state

def detect_patterns(state: DataState) -> DataState:
    df = state["df"].copy()
    features = df[[col for col in df.columns if "Q-" in col or "S-" in col]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df["PCA1"] = pca_result[:, 0]
    df["PCA2"] = pca_result[:, 1]

    kmeans = KMeans(n_clusters=3, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    state["processed_df"] = df
    return state

import plotly.graph_objects as go
import pandas as pd

def visualize(state: DataState) -> DataState:
    df = state["processed_df"]
    st.subheader("Visualisations")

    # S'assurer que "Date" est bien de type datetime
    df["Date"] = pd.to_datetime(df["Date"])
    df["Année"] = df["Date"].dt.year

    # Filtrer les 5 dernières années disponibles
    dernieres_annees = sorted(df["Année"].unique())[-5:]
    df_filtré = df[df["Année"].isin(dernieres_annees)]

    sales_cols = [col for col in df.columns if col.startswith("S-")]
    if sales_cols:
        for col in sales_cols:
            st.markdown(f"### Histogramme annuel des ventes pour {col}")
            
            # Regroupement par année
            ventes_annuelles = df_filtré.groupby("Année")[col].sum().reset_index()

            # Plot
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=ventes_annuelles["Année"],
                y=ventes_annuelles[col],
                name=col,
                marker_color='teal'
            ))

            fig.update_layout(
                title=f"Histogramme des ventes par an pour {col} (5 dernières années)",
                xaxis_title="Année",
                yaxis_title="Total des ventes",
                bargap=0.3,
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)


    # 2. Heatmap de corrélation
    st.subheader("Heatmap de Corrélation")
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax1)
    st.pyplot(fig1)

    # 3. Visualisation des clusters avec PCA
    st.subheader("Clusters (PCA)")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="tab10", ax=ax2)
    st.pyplot(fig2)

    return state

def recommend(state: DataState) -> DataState:
    df = state["processed_df"]
    avg_sales = df[[col for col in df.columns if "S-" in col]].mean()
    best = avg_sales.idxmax()
    st.success(f"Produit avec meilleures ventes moyennes : {best} ({avg_sales[best]:.2f})")
    return state

# ------------ 3. LangGraph Construction ------------

builder = StateGraph(DataState)
builder.add_node("Load Data", load_data)
builder.add_node("Inspect", inspect_data)
builder.add_node("Detect", detect_patterns)
builder.add_node("Visualize", visualize)
builder.add_node("Recommend", recommend)

builder.set_entry_point("Load Data")
builder.add_edge("Load Data", "Inspect")
builder.add_edge("Inspect", "Detect")
builder.add_edge("Detect", "Visualize")
builder.add_edge("Visualize", "Recommend")

graph = builder.compile()

# ------------ 4. Streamlit App ------------
st.title("📊 Analyse datasets des Produits")

uploaded_file = st.file_uploader("📁 Charger un fichier CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if st.button("🚀 Lancer l'analyse"):
        initial_state = {"df": df}
        graph.invoke(initial_state)
#python -m streamlit run app.py