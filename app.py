import streamlit as st
from core.llm_engine import configure_llm
from components.steps import render_step_1, render_step_2, render_step_3, render_step_4

st.set_page_config(page_title="HITL ML Studio", layout="wide")

# Initialize State
if 'step' not in st.session_state:
    st.session_state.step = 1

def main():
    st.title("⚙️ HITL Machine Learning Studio")
    configure_llm() # Authenticate API
    
    # State Machine Router
    if st.session_state.step == 1:
        render_step_1()
    elif st.session_state.step == 2:
        render_step_2()
    elif st.session_state.step == 3:
        render_step_3()
    elif st.session_state.step == 4:
        render_step_4()

if __name__ == "__main__":
    main()