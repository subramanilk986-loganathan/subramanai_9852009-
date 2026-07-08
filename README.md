    # Heart Disease Prediction with LangChain (End-to-End)

    This project demonstrates a simple end-to-end pipeline:
    - Train a RandomForest model on a heart disease dataset.
    - Expose a prediction function that accepts feature dicts.
    - Provide a Streamlit and CLI interfaces.
    - Include a LangChain Tool wrapper (requires OpenAI key) to build a conversational agent.

    ## Project structure
    [
  "data",
  "notebooks",
  "models",
  "src",
  "app",
  "requirements.txt"
]

    ## Quick start (local)
    1. Create a virtualenv and install requirements:
       ```bash
       pip install -r requirements.txt
       ```
    2. Train model (this script will also run during packaging):
       ```bash
       python models/train_model.py
       ```
    3. Run CLI:
       ```bash
       python app/main_cli.py
       ```
       OR run Streamlit app:
       ```bash
       streamlit run app/main_streamlit.py
       ```
    4. LangChain agent:
       - Set `OPENAI_API_KEY` in your environment.
       - Import `create_agent` from `src.langchain_integration` and call it.

    ## Notes
    - `data/heart_disease.csv` is included (the dataset you uploaded).
    - The model and columns are saved to `models/heart_model.pkl` by the training script.
    - The LangChain integration requires an OpenAI key and internet access.

\nModel test accuracy: 0.8065\nClassification report:\n{'No': {'precision': 0.8065, 'recall': 1.0, 'f1-score': 0.8928867976750623, 'support': 1613}, 'Yes': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 387}, 'accuracy': 0.8065, 'macro avg': {'precision': 0.40325, 'recall': 0.5, 'f1-score': 0.44644339883753115, 'support': 2000}, 'weighted avg': {'precision': 0.65044225, 'recall': 0.8065, 'f1-score': 0.7201132023249378, 'support': 2000}}\n