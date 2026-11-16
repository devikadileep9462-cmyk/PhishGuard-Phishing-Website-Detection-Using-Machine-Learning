
import streamlit as st
import pandas as pd
import joblib

from urllib.parse import urlparse
from url_features import extract_url_features


# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="PhishGuard",
    page_icon="🔐",
    layout="centered"
)


# --------------------------------
# Load Model
# --------------------------------

@st.cache_resource
def load_model():

    bundle = joblib.load("phishguard_model.pkl")

    return bundle


try:
    bundle = load_model()

    model = bundle["model"]
    feature_names = bundle["feature_names"]

except Exception:
    st.error(
        "Model could not be loaded. "
        "Please run train_url_model.py first."
    )
    st.stop()


# --------------------------------
# Custom CSS
# --------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: #16325c;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 16px;
}

.stButton>button {
    width: 100%;
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 48px;
    font-size: 17px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# Header
# --------------------------------

st.markdown(
    '<div class="title">🔐 PhishGuard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Website Security Checker'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

st.info(
    "Check a website address for URL patterns "
    "commonly associated with phishing."
)


# --------------------------------
# URL Input
# --------------------------------

st.subheader("Check a Website")

url = st.text_input(
    "Enter website URL",
    placeholder="https://example.com"
)


# --------------------------------
# Prediction
# --------------------------------

if st.button("🔍 Check Website"):

    if not url.strip():

        st.warning("Please enter a website URL.")

    else:

        cleaned_url = url.strip()

        if not cleaned_url.startswith(("http://", "https://")):
            cleaned_url = "https://" + cleaned_url

        parsed = urlparse(cleaned_url)

        if not parsed.hostname or "." not in parsed.hostname:

            st.error("Please enter a valid website address.")

        else:

            with st.spinner("Analyzing website address..."):

                try:

                    # Extract features
                    features = extract_url_features(cleaned_url)

                    input_df = pd.DataFrame(
                        [features],
                        columns=feature_names
                    )

                    # Predict
                    prediction = model.predict(input_df)[0]

                    probability = model.predict_proba(input_df)[0][1]

                    risk_percentage = probability * 100

                except Exception as e:

                    st.error(f"Prediction error: {e}")
                    st.stop()

            st.divider()

            st.subheader("Website Analysis Result")

            if prediction == 1:

                st.error("⚠️ Potential Phishing Website")

                st.write(
                    "This website has URL characteristics "
                    "associated with phishing websites."
                )

            else:

                st.success("✅ Likely Legitimate Website")

                st.write(
                    "The URL patterns appear less suspicious "
                    "according to the trained model."
                )

            # Risk score
            st.subheader("Estimated Phishing Risk")

            st.metric(
                "Model-estimated risk",
                f"{risk_percentage:.2f}%"
            )

            st.progress(float(probability))

            st.caption(
                "This score is a model estimate, not a guarantee "
                "of website safety."
            )

            st.write("**Website checked:**", cleaned_url)


# --------------------------------
# Footer
# --------------------------------

st.divider()

st.caption(
    "PhishGuard | Machine Learning Based Website Security"
)

st.caption(
    "This tool analyzes URL patterns only. "
    "It does not inspect the live webpage or guarantee safety. "
    "Never enter sensitive information on a suspicious website."
)
