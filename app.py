import streamlit as st
import joblib
import pandas as pd
from os.path import dirname, join, realpath
import joblib

# add banner image
st.header("Financial Inclusion in Africa")
st.image("images/financial-inclusion.jpeg")
st.subheader(
    """
A simple app to predict who in Africa is most likely to have a bank account?.
"""
)

# form to collect news content
my_form = st.form(key="financial_form")
country = my_form.selectbox("select country", ("Tanzania", "kenya", "Uganda", "Rwanda"))
location_type = my_form.selectbox("select location", ("Rural", "Urban"))
year = my_form.number_input("Inter year", min_value=2000, max_value=2100)
cellphone_access = my_form.selectbox("Do you have a cellphone?", ("Yes", "No"))
gender_of_respondent = my_form.selectbox("Gender", ("Female", "Male"))
relationship_with_head = my_form.selectbox(
    "what is your relationship with the head of the family",
    (
        "Spouse",
        "Head of Household",
        "Other relative",
        "Child",
        "Parent",
        "Other non-relatives",
    ),
)
marital_status = my_form.selectbox(
    "Your marital status",
    (
        "Married/Living together",
        "Widowed",
        "Single/Never Married",
        "Divorced/Seperated",
        "Dont know",
    ),
)
education_level = my_form.selectbox(
    "Your education level",
    (
        "Secondary education",
        "No formal education",
        "Vocation/Specialised training",
        "Primary education",
        "Tertiary education",
        "Other/Dont know/RTA",
    ),
)
job_type = my_form.selectbox(
    "Your job type",
    (
        "Self employed",
        "Government Dependent",
        "Formally employed Private",
        "Informally employed",
        "Formally employed Government",
        "Farming and Fishing",
        "Remittance Dependent",
        "Other Income",
        "Dont Know/Refuse to answer",
        "No Income",
    ),
)
household_size = my_form.number_input(
    "How many people are living in the house?", min_value=1, max_value=100
)

age_of_respondent = my_form.number_input("Your age", min_value=18, max_value=120)

submit = my_form.form_submit_button(label="make prediction")


# load the model and one-hot-encoder and scaler

with open(
    join(dirname(realpath(__file__)), "model/lightgbm-financial-inclusion-model.pkl"),
    "rb",
) as f:
    model = joblib.load(f)

with open(
    join(dirname(realpath(__file__)), "preprocessing/min-max-scaler.pkl"), "rb"
) as f:
    scaler = joblib.load(f)


with open(
    join(dirname(realpath(__file__)), "preprocessing/one-hot-encoder.pkl"), "rb"
) as f:
    one_hot_encoder = joblib.load(f)


sentiments = {0: "Neutral", 1: "Positive", -1: "Negative"}


@st.cache
# function to clean and tranform the input
def preprocessing_data(data, enc, scaler):

    # Convert the following numerical labels from integer to float
    float_array = data[["household_size", "age_of_respondent", "year"]].values.astype(
        float
    )

    # One Hot Encoding conversion
    data = enc.transform(data)

    # scale our data into range of 0 and 1
    data = scaler.transform(data)

    return data


if submit:

    # collect inputs
    input = {
        "country": country,
        "year": year,
        "location_type": location_type,
        "cellphone_access": cellphone_access,
        "household_size": household_size,
        "age_of_respondent": age_of_respondent,
        "gender_of_respondent": gender_of_respondent,
        "relationship_with_head": relationship_with_head,
        "marital_status": marital_status,
        "education_level": education_level,
        "job_type": job_type,
    }

    # create a draframe
    data = pd.DataFrame(input, index=[0])

    # clean and transform input
    transformed_data = preprocessing_data(data=data, enc=one_hot_encoder, scaler=scaler)

    # perform prediction
    prediction = model.predict(transformed_data)
    output = int(prediction[0])
    probas = model.predict_proba(transformed_data)
    probability = "{:.2f}".format(float(probas[:, output]))

    # Display results of the NLP task
    st.header("Results")
    if output == 1:
        st.write(
            "You are most likely to have a bank account with probability of {} 😊".format(
                probability
            )
        )
    elif output == 0:
        st.write(
            "You are most likely not to have a bank account with probability of {} 😔".format(
                probability
            )
        )


st.write("Developed with ❤️ by Davis David")
