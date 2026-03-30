import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

# Title
st.title("📩 SMS Spam Classification using Lasso (Feature Selection)")

# Upload dataset
uploaded_file = st.file_uploader("Upload SMS Spam Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')

    # Keep only necessary columns
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    # Convert labels to numeric
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['message'])
    y = df['label']

    total_features = X.shape[1]
    st.write(f"### Total TF-IDF Features: {total_features}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Function to train Lasso
    def run_lasso(alpha):
        model = Lasso(alpha=alpha)
        model.fit(X_train, y_train)

        coeffs = model.coef_
        non_zero = np.sum(coeffs != 0)
        zero = np.sum(coeffs == 0)

        return non_zero, zero, coeffs

    st.write("## 🔍 Lasso Results")

    # Alpha = 0.1
    nz_01, z_01, coef_01 = run_lasso(0.1)

    st.write("### Alpha = 0.1")
    st.write(f"Non-zero coefficients: {nz_01}")
    st.write(f"Eliminated features: {z_01}")

    # Alpha = 0.01
    nz_001, z_001, _ = run_lasso(0.01)

    st.write("### Alpha = 0.01")
    st.write(f"Selected features: {nz_001}")

    # Alpha = 1
    nz_1, z_1, _ = run_lasso(1)

    st.write("### Alpha = 1")
    st.write(f"Selected features: {nz_1}")

    # Percentage reduction
    reduction = ((total_features - nz_01) / total_features) * 100

    st.write("## 📉 Feature Reduction")
    st.write(f"Reduction Percentage (alpha=0.1): {reduction:.2f}%")

    # Show top important words
    feature_names = vectorizer.get_feature_names_out()
    important_idx = np.where(coef_01 != 0)[0]

    important_words = feature_names[important_idx]

    st.write("## ⭐ Important Selected Words")
    st.write(important_words[:20])  # show top 20
