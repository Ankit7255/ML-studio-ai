# ML-Studio-AI: Human-in-the-Loop Machine Learning Environment

## Overview
ML-Studio-AI is a modular, interactive web application designed to bridge the gap between fully automated machine learning pipelines and manual data science workflows. By implementing a Human-in-the-Loop (HITL) architecture, the application leverages Large Language Models (LLMs) to analyze dataset metadata and recommend preprocessing strategies, while allowing the human domain expert to make the final architectural decisions.

## Architecture
The system is built on a component-driven architecture to ensure scalability and strict separation of concerns:
* **`app.py`**: The state machine and main routing gateway.
* **`core/llm_engine.py`**: Manages secure API communication, parsing dataset metadata (cardinality, dtypes, null distributions) to generate context-aware preprocessing strategies without exposing raw row data.
* **`core/ml_engine.py`**: Handles dynamic one-hot encoding, train/test splitting, and Scikit-Learn model training (Classification/Regression).
* **`components/steps.py`**: Contains the decoupled UI views for the multi-step workflow.

## Key Features
* **Metadata-Driven LLM Analysis**: Utilizes Google Gemini 1.5 Flash to analyze statistical metadata and recommend specific feature engineering steps (e.g., flagging high-cardinality identifiers for removal to prevent overfitting).
* **Zero Data-Loss Imputation**: Provides interactive UI controls to impute missing numerical values (median) and categorical values ('Unknown') to preserve dataset integrity before model training.
* **Dynamic Model Builder**: Automatically detects feature sets, handles categorical encoding, and trains baseline Random Forest models with real-time performance metrics (Accuracy/MSE).
* **Live Inference Engine**: Dynamically generates input forms based on the finalized training features, allowing for immediate single-row predictions on unseen data.

## Technology Stack
* **Frontend & State Management:** Streamlit
* **Data Manipulation:** Pandas
* **Machine Learning:** Scikit-Learn
* **LLM Orchestration:** Google Generative AI (Gemini API)

## Local Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/Ankit7255/ML-studio-ai.git](https://github.com/Ankit7255/ML-studio-ai.git)
   cd ML-studio-ai


# Live Deployment
The application is deployed via Streamlit Community Cloud and features CI/CD integration with the main branch of this repository.

Developed by Ankit as a scalable portfolio project demonstrating end-to-end ML engineering, modular component architecture, and LLM API integration.