import streamlit as st
import pickle
import pandas as pd

# Load models
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Diabetes Prediction
def diabetes_prediction(input_data):
    model = load_model('diabetes_model.pkl')
    prediction = model.predict(input_data)
    return prediction

# Thyroid Prediction
def thyroid_prediction(input_data):
    model = load_model('thyroid_model.pkl')
    prediction = model.predict(input_data)
    return prediction

# Lung Cancer Prediction
def lung_cancer_prediction(input_data):
    model = load_model('lung_cancer_model.pkl')
    prediction = model.predict(input_data)
    return prediction

# Parkinson's Prediction
def parkinsons_prediction(input_data):
    model = load_model('parkinsons_model.pkl')
    prediction = model.predict(input_data)
    return prediction

# Main App
def main():
    st.title("AI-Powered Medical Diagnosis")
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Choose a Disease", ["Diabetes", "Thyroid", "Lung Cancer", "Parkinson's"])

    if choice == "Diabetes":
        st.header("Diabetes Prediction")
        pregnancies = st.number_input("Number of Pregnancies", 0, 17)
        glucose = st.number_input("Glucose Level", 0, 200)
        blood_pressure = st.number_input("Blood Pressure", 0, 122)
        skin_thickness = st.number_input("Skin Thickness", 0, 99)
        insulin = st.number_input("Insulin Level", 0, 846)
        bmi = st.number_input("BMI", 0.0, 67.1)
        diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", 0.078, 2.42)
        age = st.number_input("Age", 21, 81)

        input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]],
                                 columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

        if st.button("Predict"):
            prediction = diabetes_prediction(input_data)
            if prediction[0] == 1:
                st.error("The model predicts that you have Diabetes.")
            else:
                st.success("The model predicts that you do not have Diabetes.")

    elif choice == "Thyroid":
        st.header("Thyroid Disorder Prediction")
        # Add input fields for thyroid disorder prediction
        # Example: age, sex, TSH, T3, TT4, etc.
        # input_data = ...

        if st.button("Predict"):
            prediction = thyroid_prediction(input_data)
            if prediction[0] == 1:
                st.error("The model predicts that you have a Thyroid Disorder.")
            else:
                st.success("The model predicts that you do not have a Thyroid Disorder.")

    elif choice == "Lung Cancer":
        st.header("Lung Cancer Prediction")
        # Add input fields for lung cancer prediction
        # Example: age, smoking, yellow_fingers, anxiety, etc.
        # input_data = ...

        if st.button("Predict"):
            prediction = lung_cancer_prediction(input_data)
            if prediction[0] == 1:
                st.error("The model predicts that you have Lung Cancer.")
            else:
                st.success("The model predicts that you do not have Lung Cancer.")

    elif choice == "Parkinson's":
        st.header("Parkinson's Disease Prediction")
        # Add input fields for Parkinson's disease prediction
        # Example: MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz), etc.
        # input_data = ...

        if st.button("Predict"):
            prediction = parkinsons_prediction(input_data)
            if prediction[0] == 1:
                st.error("The model predicts that you have Parkinson's Disease.")
            else:
                st.success("The model predicts that you do not have Parkinson's Disease.")

if __name__ == "__main__":
    main()