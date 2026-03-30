import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Title
st.title("📩 SMS Spam Classification with L1 (Lasso) Feature Selection")

# Upload dataset
uploaded_file = st.file_uploader("Upload SMS Spam Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')

    # Keep required columns
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    # Convert labels
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    st.write("### 📊 Dataset Preview")
    st.dataframe(df.head())

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['message'])
    y = df['label']

    total_features = X.shape[1]
    st.write(f"### 🔢 Total TF-IDF Features: {total_features}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Function using Logistic Regression (L1)
    def run_l1_model(alpha):
        model = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=1/alpha,
            max_iter=1000
        )
        
        model.fit(X_train, y_train)

        coeffs = model.coef_[0]

        non_zero = np.sum(coeffs != 0)
        zero = np.sum(coeffs == 0)

        return non_zero, zero, coeffs

    st.write("## 🔍 Feature Selection Results")

    # Alpha = 0.1
    nz_01, z_01, coef_01 = run_l1_model(0.1)
    st.write("### Alpha = 0.1")
    st.write(f"✅ Selected Features: {nz_01}")
    st.write(f"❌ Eliminated Features: {z_01}")

    # Alpha = 0.01
    nz_001, z_001, coef_001 = run_l1_model(0.01)
    st.write("### Alpha = 0.01")
    st.write(f"Selected Features: {nz_001}")

    # Alpha = 1
    nz_1, z_1, coef_1 = run_l1_model(1)
    st.write("### Alpha = 1")
    st.write(f"Selected Features: {nz_1}")

    # Reduction %
    reduction = ((total_features - nz_01) / total_features) * 100

    st.write("## 📉 Feature Reduction")
    st.write(f"Reduction Percentage (alpha=0.1): {reduction:.2f}%")

    # Extract important words
    feature_names = vectorizer.get_feature_names_out()
    important_idx = np.where(coef_01 != 0)[0]

    st.write("## ⭐ Important Selected Words")

    if len(important_idx) == 0:
        st.warning("⚠️ No features selected. Try smaller alpha.")
    else:
        important_words = feature_names[important_idx]

        # Show top 20 words
        st.write(important_words[:20])

    # Optional: User input prediction
    st.write("## ✉️ Try Your Own Message")

    user_input = st.text_area("Enter SMS message:")

    if st.button("Predict"):
        model = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=1/0.1,
            max_iter=1000
        )
        model.fit(X_train, y_train)

        user_vec = vectorizer.transform([user_input])
        pred = model.predict(user_vec)[0]

        if pred == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam (Ham)")
