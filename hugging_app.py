import streamlit as st
from rdkit import Chem
from mordred import Calculator, descriptors
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="SMILES Taste Predictor", page_icon="🍬")
st.title("🍭 SMILES Taste Classifier (Sweet vs. Bitter)")

smiles = st.text_input("Enter SMILES:")
model_name = st.selectbox("Select Model", ["XGB", "RF", "MLP", "LR", "VOTE"])

if st.button("Predict") and smiles:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("Invalid SMILES structure.")
        else:
            desc_calc = Calculator(descriptors, ignore_3D=True)
            df = desc_calc.pandas([mol]).replace([np.inf, -np.inf], np.nan)
            X = df.select_dtypes(include=[np.number])

            descriptor_columns = joblib.load("descriptor_columns.pkl")
            for col in descriptor_columns:
                if col not in X.columns:
                    X[col] = np.nan
            X = X[descriptor_columns].fillna(X.mean()).fillna(0)

            scaler = joblib.load("scaler.pkl")
            X_scaled = scaler.transform(X.values.astype(np.float32))

            model_file = "voting_model.pkl" if model_name == "VOTE" else f"{model_name.lower()}_model.pkl"
            model = joblib.load(model_file)

            prediction = model.predict(X_scaled)[0]
            st.success("🍬 Sweet!" if prediction == 1 else "🍋 Bitter!")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
