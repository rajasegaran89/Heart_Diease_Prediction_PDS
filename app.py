import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import pickle
import sklearn


DATASET_PATH = "data/heart_2020_cleaned.csv"
LOG_MODEL_PATH = "model/logistic_regression.pkl"


def main():
    @st.cache(persist=True)
    def load_dataset() -> pd.DataFrame:
        heart_df = pl.read_csv(DATASET_PATH)
        heart_df = heart_df.to_pandas()
        heart_df = pd.DataFrame(np.sort(heart_df.values, axis=0),
                                index=heart_df.index,
                                columns=heart_df.columns)
        return heart_df


    def user_input_features() -> pd.DataFrame:
        race = st.sidebar.selectbox("Race", options=(race for race in heart.Race.unique()))
        sex = st.sidebar.selectbox("Sex", options=(sex for sex in heart.Sex.unique()))
        age_cat = st.sidebar.selectbox("Age category",
                                       options=(age_cat for age_cat in heart.AgeCategory.unique()))
        bmi_cat = st.sidebar.selectbox("BMI category",
                                       options=(bmi_cat for bmi_cat in heart.BMICategory.unique()))
        sleep_time = st.sidebar.number_input("How many hours on average do you sleep?", 0, 24, 7)
        gen_health = st.sidebar.selectbox("How can you define your general health?",
                                          options=(gen_health for gen_health in heart.GenHealth.unique()))
        phys_health = st.sidebar.number_input("For how many days during the past 30 days was"
                                              " your physical health not good?", 0, 30, 0)
        ment_health = st.sidebar.number_input("For how many days during the past 30 days was"
                                              " your mental health not good?", 0, 30, 0)
        phys_act = st.sidebar.selectbox("Have you played any sports (running, biking, etc.)"
                                        " in the past month?", options=("No", "Yes"))
        smoking = st.sidebar.selectbox("Have you smoked at least 100 cigarettes in"
                                       " your entire life (approx. 5 packs)?)",
                                       options=("No", "Yes"))
        alcohol_drink = st.sidebar.selectbox("Do you have more than 14 drinks of alcohol (men)"
                                             " or more than 7 (women) in a week?", options=("No", "Yes"))
        stroke = st.sidebar.selectbox("Did you have a stroke?", options=("No", "Yes"))
        diff_walk = st.sidebar.selectbox("Do you have serious difficulty walking"
                                         " or climbing stairs?", options=("No", "Yes"))
        diabetic = st.sidebar.selectbox("Have you ever had diabetes?",
                                        options=(diabetic for diabetic in heart.Diabetic.unique()))
        asthma = st.sidebar.selectbox("Do you have asthma?", options=("No", "Yes"))
        kid_dis = st.sidebar.selectbox("Do you have kidney disease?", options=("No", "Yes"))
        skin_canc = st.sidebar.selectbox("Do you have skin cancer?", options=("No", "Yes"))

        features = pd.DataFrame({
            "PhysicalHealth": [phys_health],
            "MentalHealth": [ment_health],
            "SleepTime": [sleep_time],
            "BMICategory": [bmi_cat],
            "Smoking": [smoking],
            "AlcoholDrinking": [alcohol_drink],
            "Stroke": [stroke],
            "DiffWalking": [diff_walk],
            "Sex": [sex],
            "AgeCategory": [age_cat],
            "Race": [race],
            "Diabetic": [diabetic],
            "PhysicalActivity": [phys_act],
            "GenHealth": [gen_health],
            "Asthma": [asthma],
            "KidneyDisease": [kid_dis],
            "SkinCancer": [skin_canc]
        })

        return features


    st.set_page_config(
        page_title="Heart Disease Prediction App",
        page_icon="images/heart-fav.png"
    )
    st.info("22054606 Basubeit, Omar Gumaan Saleh")
    st.text("22050516 Chandra Mohan a/l Rajendran")
    st.text("22052733 Rajasegaran a/l M Sivaanandan")
    st.text("22051081 Tan Kai Ying")
    st.text("S2190151 Wee Hin Sheik")
    st.title("Data Science Application in Predicting Heart Disease")
    st.subheader("Are you wondering about the condition of your heart? "
                 "This app will help you to diagnose it!")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image("images/doctor_new.png",
                 caption="I'll help you diagnose your heart health! - Dr. Logistic Regression",
                 width=150)
        st.markdown('##')
        m = st.markdown("""
            <style>
            div.stButton > button:first-child {
            background-color: #00cc00;color:white;
            font-size:15px;height:4em;width:20em;
            }
            </style>""", unsafe_allow_html=True)
        submit = st.button("Predict")
    with col2:
        st.markdown("""
        Is your heart healthy? Did you know that heart attacks are the leading cause of death globally? 
        An estimated 17.9 million people died from CVDs in 2019.Therefore, we highly encourage you to have a heart disease test. 
        This app calculates the probability of an individual based on the parameter given to predict heart disease within seconds. 
        We are calculating the probability based on the survey data of over 300 thousand United States residents from 2020. 
        It will provide 80% accuracy of the result. Here are a few steps away to predicting your heart disease status:
            1.    Enter the parameters based on your health status.
            2.    Press the "Predict" button and wait seconds to get the result. 
        Please remember that this result does not constitute a diagnosis from a doctor!   
        Due to this model's imperfect accuracy, healthcare facilities would never employ it. 
        Therefore, if you experience any concerns or problems, see a doctor.
            """)

    heart = load_dataset()

    st.sidebar.title("Feature Selection")
    st.sidebar.image("images/heart-sidebar.png", width=100)


    input_df = user_input_features()
    df = pd.concat([input_df, heart], axis=0)
    df = df.drop(columns=["HeartDisease"])

    cat_cols = ["BMICategory", "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking",
                "Sex", "AgeCategory", "Race", "Diabetic", "PhysicalActivity",
                "GenHealth", "Asthma", "KidneyDisease", "SkinCancer"]
    for cat_col in cat_cols:
        dummy_col = pd.get_dummies(df[cat_col], prefix=cat_col)
        df = pd.concat([df, dummy_col], axis=1)
        del df[cat_col]

    df = df[:1]
    df.fillna(0, inplace=True)

    log_model = pickle.load(open(LOG_MODEL_PATH, "rb"))

    if submit:
        prediction = log_model.predict(df)
        prediction_prob = log_model.predict_proba(df)
        if prediction == 0:
            st.markdown(f"**The probability that you'll have"
                        f" heart disease is {round(prediction_prob[0][1] * 100, 2)}%."
                        f" You are healthy!**")
            st.image("images/heart-okay_new.jpg",
                     caption="Your heart seems to be okay! - Dr. Logistic Regression")
            st.image("images/Age_Category.png")
            st.image("images/Pearson.jpg")
        else:
            st.markdown(f"**The probability that you will have"
                        f" heart disease is {round(prediction_prob[0][1] * 100, 2)}%."
                        f" It sounds like you are not healthy.**")
            st.image("images/heart-bad_new.jpg",
                     caption="I'm not satisfied with the condition of your heart! - Dr. Logistic Regression")
            st.image("images/Age_Category.png")
            st.image("images/Pearson.jpg")


if __name__ == "__main__":
    main()
