import streamlit as st
import joblib
import pandas as pd

st.title("Irisian Sumit 🌸")
st.write("Upload a CSV with 4 iris features to get predictions.")

# Load model
modelRFPreTrained = joblib.load("rfTrainedModel.pkl")

# Upload CSV
uploaded_file = st.file_uploader("Upload test data", type="csv")

# Proceed only if a file is uploaded
if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)
        predicted_data = modelRFPreTrained.predict(test_df)
        pred_df = pd.DataFrame(predicted_data, columns=["prediction"])

        st.download_button(
            label="Download Predictions",
            data=pred_df.to_csv(index=False),
            file_name="iris_predictions.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
else:
    st.info("📥 Please upload a CSV file to begin.")
