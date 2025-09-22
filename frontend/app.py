import streamlit as st
import pandas as pd
import sys, os
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from langgraph_flow.langgraph_pipeline.workflow import build_pipeline
from langgraph_flow.agents.visualize_data import visualize
from langgraph_flow.agents.detect_patterns import detect_patterns
from langgraph_flow.agents.generate_rapport import generate_report

st.title("Product Datasets Analysis")

# Upload du fichier
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state['df'] = df
    st.subheader('File uploaded')

# Vérifier si le fichier est uploadé
if 'df' in st.session_state:
    df = st.session_state['df']

    if st.button("Run Analysis"):
        pipeline = build_pipeline()
        initial_state = {"df": df}
        state = pipeline.invoke(initial_state)
        st.session_state['state'] = state

        cleaned_df = state["processed_df"]

        st.subheader("Dataset Dimensions")
        st.write(f"Before cleaning: {df.shape}")
        st.write(f"After cleaning: {cleaned_df.shape}")

        st.subheader("Missing Values")
        st.write("Before cleaning:", df.isnull().sum())
        st.write("After cleaning:", cleaned_df.isnull().sum())

        st.subheader("Descriptive Statistics")
        st.write(cleaned_df.describe())

        st.subheader("Visualizations")
        figs = visualize(state)

        # Histogrammes
        for fig in figs["hist"]:
            st.plotly_chart(fig)

        # Heatmap
        st.subheader("Correlation Matrix")
        if figs["heatmap"] is not None:
            st.pyplot(figs["heatmap"])

        # Détection des clusters
        state = detect_patterns(state)
        st.session_state['state'] = state

        pca_df = state['pca_df']
        st.subheader("Clusters (PCA)")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=pca_df, x="PCA1", y="PCA2", hue="Cluster", palette="tab10", ax=ax2)
        st.pyplot(fig2)

# Génération du rapport et bouton download
if 'state' in st.session_state:
    state = st.session_state['state']

    if st.button('Generate Report'):
        state = generate_report(state)
        st.session_state['state'] = state
        st.session_state['report'] = state['report']

    if 'report' in st.session_state:
        report = st.session_state['report']
        st.download_button(
            label="Download Report",
            data=report,
            file_name="report.txt",
            mime="text/plain"
        )
