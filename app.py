import streamlit as st
import pickle

# Load trained pipeline (TF-IDF + RandomForest)
model = pickle.load(open("spam_pipeline.pkl", "rb"))

# App title
st.title("📩 Spam Message Classifier")
st.write("Enter a **preprocessed message** and the model will predict if it's Spam (1) or Not Spam (0).")

# Input box
message = st.text_input("Enter preprocessed message:", "")

# Prediction function
def predict_message(msg):
    result = model.predict([msg])[0]
    return "Spam (1)" if result == 1 else "Not Spam (0)"

# Predict button
if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message first!")
    else:
        prediction = predict_message(message)
        st.success(f"Prediction: **{prediction}**")

# Footer with your name
st.markdown("""
---
### 👨‍💻 Developed by **Om Laxman Khairnar**
""")
