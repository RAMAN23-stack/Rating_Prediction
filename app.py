import streamlit as st
import pandas as pd
import joblib

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("rating_prediction_model.pkl")

model = load_model()

st.set_page_config(page_title="Rating Prediction App", layout="centered")

st.title("⭐ Rating Prediction App")
st.write("Enter the restaurant details to predict its aggregate rating.")

# Load dataset to get feature structure
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset .csv")
    df = df.dropna()
    # Note: Aggregate rating is our target
    return df

df = load_data()

# Separate Numerical and Categorical columns for the UI
numerical_cols = ['Average Cost for two', 'Price range', 'Votes', 'Longitude', 'Latitude']
# Aggregate rating is the target, we don't input it

st.subheader("Input Features")

# Create inputs for main numerical features
user_input_values = {}
col1, col2 = st.columns(2)

with col1:
    user_input_values['Average Cost for two'] = st.number_input("Average Cost for two", value=float(df['Average Cost for two'].mean()))
    user_input_values['Price range'] = st.number_input("Price range (1-4)", value=float(df['Price range'].mode()[0]), min_value=1.0, max_value=4.0)
    user_input_values['Votes'] = st.number_input("Votes", value=float(df['Votes'].mean()))

with col2:
    user_input_values['Longitude'] = st.number_input("Longitude", value=float(df['Longitude'].mean()))
    user_input_values['Latitude'] = st.number_input("Latitude", value=float(df['Latitude'].mean()))
    user_input_values['Country Code'] = st.number_input("Country Code", value=float(df['Country Code'].mode()[0]))
    user_input_values['Restaurant ID'] = st.number_input("Restaurant ID", value=float(df['Restaurant ID'].mode()[0]))

if st.button("Predict Rating"):
    # 1. Prepare the full feature set (matching the 20,807 dummies)
    # We load the dummy structure from a small sample to avoid memory issues
    df_sample = df.head(10)
    df_encoded_cols = pd.get_dummies(df, drop_first=True).drop("Aggregate rating", axis=1).columns
    
    # Initialize a DataFrame with zeros for all features
    input_df = pd.DataFrame(0, index=[0], columns=df_encoded_cols)
    
    # Fill in the numerical values provided by the user
    for col, val in user_input_values.items():
        if col in input_df.columns:
            input_df.at[0, col] = val
            
    # Note: Categorical dummies remain 0 (the 'reference' category or mean/mode behavior)
    # This keeps the app responsive while remaining compatible with the model
    
    prediction = model.predict(input_df)
    st.success(f"### Predicted Aggregate Rating: {prediction[0]:.2f}")
    
    # Show some context
    st.info(f"The average rating in the dataset is {df['Aggregate rating'].mean():.2f}")
