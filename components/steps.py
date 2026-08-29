import streamlit as st
import pandas as pd
from core.llm_engine import get_preprocessing_suggestions
from core.ml_engine import train_model

def render_step_1():
    st.header("Step 1: Upload Dataset")
    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        st.session_state.df = pd.read_csv(file)
        st.dataframe(st.session_state.df.head())
        if st.button("Next: Preprocessing Studio"):
            st.session_state.step = 2
            st.rerun()

def render_step_2():
    st.header("Step 2: AI Preprocessing & Pruning")
    
    # 1. Dataset Status Tracker
    st.write(f"**Current Dataset:** `{st.session_state.df.shape[0]}` rows | `{st.session_state.df.shape[1]}` columns")
    st.dataframe(st.session_state.df.head())
    
    # 2. AI Suggestions Generator
    if st.button("Generate AI Suggestions"):
        with st.spinner("LLM is analyzing dataset metadata..."):
            st.session_state.suggestions = get_preprocessing_suggestions(st.session_state.df)
            
    if st.session_state.get('suggestions'):
        st.subheader("AI Recommendations:")
        st.markdown(st.session_state.suggestions)
        
    st.divider()
    st.subheader("Action Panel")
    
    col_act1, col_act2 = st.columns(2)
    
    # 3. Missing Values Action (UPGRADED)
    with col_act1:
        st.markdown("#### 1. Handle Missing Values")
        
        # Show exactly which columns have nulls
        null_counts = st.session_state.df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        
        if not cols_with_nulls.empty:
            st.write("Columns missing data:")
            st.dataframe(cols_with_nulls)
            
            # Action: Impute Numerical Data
            num_cols_with_nulls = st.session_state.df.select_dtypes(include=['number']).columns.intersection(cols_with_nulls.index)
            if len(num_cols_with_nulls) > 0:
                if st.button("Fill Numerical Nulls with Median"):
                    for col in num_cols_with_nulls:
                        st.session_state.df[col] = st.session_state.df[col].fillna(st.session_state.df[col].median())
                    st.success("Filled missing numbers with median!")
                    st.rerun()
            
            # Action: Impute Categorical Data
            cat_cols_with_nulls = st.session_state.df.select_dtypes(exclude=['number']).columns.intersection(cols_with_nulls.index)
            if len(cat_cols_with_nulls) > 0:
                if st.button("Fill Categorical Nulls with 'Unknown'"):
                    for col in cat_cols_with_nulls:
                        st.session_state.df[col] = st.session_state.df[col].fillna('Unknown')
                    st.success("Filled missing text with 'Unknown'!")
                    st.rerun()
                    
            # The Nuclear Option (Keep it as a last resort)
            if st.button("Nuclear: Drop Rows (Not Recommended)"):
                st.session_state.df = st.session_state.df.dropna()
                st.warning("Dropped rows. Check dataset size!")
                st.rerun()
        else:
            st.success("No missing values found!")

    # 4. Universal Column Drop Action
    with col_act2:
        st.markdown("#### 2. Drop Useless Columns")
        cols_to_drop = st.multiselect(
            "Select columns to DROP (IDs, text, high-cardinality):", 
            st.session_state.df.columns
        )
        if st.button("Drop Selected Columns"):
            if cols_to_drop:
                st.session_state.df = st.session_state.df.drop(columns=cols_to_drop)
                st.success(f"Dropped {len(cols_to_drop)} columns!")
                st.rerun()
            else:
                st.warning("Select at least one column from the dropdown first.")
            
    st.divider()
    
    # 5. Navigation
    nav_col1, nav_col2 = st.columns(2)
    if nav_col1.button("Back"):
        st.session_state.step = 1
        st.rerun()
    if nav_col2.button("Next: Model Builder"):
        st.session_state.step = 3
        st.rerun()

def render_step_3():
    st.header("Step 3: Model Builder")
    df = st.session_state.df
    
    target = st.selectbox("Target Column", df.columns)
    task = st.radio("Task Type", ["Classification", "Regression"])
    
    if st.button("Train Model"):
        with st.spinner("Training ML Model..."):
            model, metric, feature_cols = train_model(df, target, task)
            st.session_state.model = model
            st.session_state.metric = metric
            st.session_state.feature_cols = feature_cols
            st.success(f"Model Trained! Performance metric: {metric:.4f}")
            
    col1, col2 = st.columns(2)
    if col1.button("Back"):
        st.session_state.step = 2
        st.rerun()
    if col2.button("Next: Inference Engine"):
        st.session_state.step = 4
        st.rerun()

def render_step_4():
    st.header("Step 4: Inference Engine")
    
    if not st.session_state.get('model'):
        st.warning("Please train a model in Step 3 first.")
        return
        
    st.write(f"**Final Model Score:** {st.session_state.metric:.4f}")
    st.subheader("Test on New Data")
    
    # Dynamically generate input fields based on trained features
    input_data = {}
    for col in st.session_state.feature_cols:
        input_data[col] = st.number_input(f"Input {col}", value=0.0)
        
    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        prediction = st.session_state.model.predict(input_df)
        st.success(f"Result: {prediction[0]}")
        st.balloons()
        
    if st.button("Start Over"):
        st.session_state.clear()
        st.session_state.step = 1
        st.rerun()