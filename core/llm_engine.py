import google.generativeai as genai
import pandas as pd
import streamlit as st

def configure_llm():
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    else:
        st.warning("Please configure your API Key in Streamlit Secrets.")

def get_preprocessing_suggestions(df: pd.DataFrame) -> str:
    # 1. Calculate cardinality for text columns so the LLM makes mathematical decisions, not guesses
    text_cols = df.select_dtypes(include=['object', 'string']).columns
    cardinality = {col: df[col].nunique() for col in text_cols}
    
    # 2. Universal metadata payload
    metadata = f"""
    Total Rows: {len(df)}
    Columns: {list(df.columns)}
    Data Types: {df.dtypes.to_dict()}
    Null Values: {df.isnull().sum().to_dict()}
    Unique Values in Text Columns: {cardinality}
    """
    
    model = genai.GenerativeModel('gemini-3.5-flash-lite')
    prompt = f"""You are a pragmatic Machine Learning Engineer advising a user on a streamlined web app. 
    Review this dataset metadata and provide 3-4 bullet points of actionable advice.

    CRITICAL UNIVERSAL RULES:
    1. IDENTIFY IDs/INDEXES: If any column has nearly as many unique values as there are total rows (especially text columns), tell the user to DROP it. It is an identifier and will cause overfitting.
    2. PREVENT ONE-HOT CRASHES: If a categorical/text column has high cardinality (too many unique string values), explicitly tell the user to DROP it. Do not suggest complex regex, date parsing, or text extraction for this MVP.
    3. NULL HANDLING: Suggest practical ways to handle the columns that have missing (null) values based on their data type.
    
    Metadata:
    {metadata}
    """
    
    response = model.generate_content(prompt)
    return response.text