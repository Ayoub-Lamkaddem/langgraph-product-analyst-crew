import streamlit as st
from typing import TypedDict
import pandas as pd
from .state import DataState

def inspect_data(state: DataState) -> DataState:
    '''df = state["df"]
    st.subheader("Aperçu des données")
    st.write(df.describe())
    st.write("Valeurs manquantes :")
    st.write(df.isnull().sum())
    st.write("Valeurs manquantes:")
    st.write(df.duplicated().sum())
    return state'''
    return
