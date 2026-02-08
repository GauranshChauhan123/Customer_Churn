import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models  import load_model
from sklearn.preprocessing import LabelEncoder,OneHotEncoder 
import pickle
import streamlit as st


st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📉 Customer Churn Prediction (ANN)")
st.write("Predict whether a customer will churn or not")

model = load_model("model.h5")

scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
ohe_encoder = pickle.load(open("ohe_encoder.pkl", "rb"))

st.subheader("Customer Information")

credit_score = st.number_input("Credit Score")
geography = st.selectbox("Geography", ohe_encoder.categories_[0])
gender = st.selectbox("Gender", label_encoder.classes_)
age = st.number_input("Age",1)
tenure = st.slider("Tenure (Years)", 0, 10, 5)
balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
num_products = st.slider("Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)



input_data = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active,
    "EstimatedSalary": estimated_salary
}
df=pd.DataFrame([input_data])
df['Gender']=label_encoder.transform(df['Gender'])
ohe=ohe_encoder.transform(df[['Geography']])
ohe.toarray()
ohe=pd.DataFrame(ohe.toarray(),columns=ohe_encoder.get_feature_names_out(['Geography']))
df=pd.concat([df.drop(['Geography'],axis=1),ohe],axis=1)

input_scaled=scaler.transform(df)




if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)
    churn_prob = prediction[0][0]

    st.subheader("Prediction Result")

    if churn_prob > 0.5:
        st.error(f"🚨 Customer is likely to churn ({churn_prob:.2%})")
    else:
        st.success(f"✅ Customer is unlikely to churn ({churn_prob:.2%})")

    st.progress(float(churn_prob))
