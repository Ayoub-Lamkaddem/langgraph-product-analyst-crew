from typing import TypedDict
from .state import DataState
from dotenv import load_dotenv
import pandas as pd
import os

# Si tu as une API Gemini LLM
from google import genai
def generate_report(state: DataState) -> str:
    
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)


    df = state.get("processed_df")
    pca_df = state.get("pca_df")

    # Préparer un résumé des données
    summary = ""
    if df is not None:
        summary += "Résumé statistique des données :\n"
        summary += df.describe().to_string() + "\n\n"
        summary += "Nombre de lignes et colonnes : {} x {}\n\n".format(df.shape[0], df.shape[1])
    
    if pca_df is not None and "Cluster" in pca_df.columns:
        clusters = pca_df["Cluster"].value_counts().to_dict()
        summary += "Répartition des clusters détectés :\n"
        for cluster, count in clusters.items():
            summary += f"Cluster {cluster} : {count} points\n"
        summary += "\n"

    # Appel à l'API Gemini pour générer un rapport plus narratif
    prompt = f"""
    Voici un résumé des données et des patterns détectés :
    {summary}
    
    Génère un rapport clair et concis pour un utilisateur non technique, expliquant :
    - les statistiques principales,
    - les patterns et clusters détectés,
    - et les insights potentiels.
    """
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )

    report_text = response.text
    # Optionnel : sauvegarder le rapport dans le state
    state["report"] = report_text
    return state
