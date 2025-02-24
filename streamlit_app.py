import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="GBR Model Predictor",
    page_icon="📊",
    layout="wide"
)

# Add title and description
st.title("GBR Model Predictor")
st.write("Enter the required features to get predictions from the GBR model.")

@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_gbr_model.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_model()

if model is not None:
    # Get required features
    required_features = model.feature_names_in_.tolist()
    
    # Create scaler
    scaler = StandardScaler()
    
    # Create sample data for scaler
    sample_data = {feature: [1, 2, 3, 4, 5] for feature in required_features}
    df = pd.DataFrame(sample_data)
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    scaler.fit(df[numeric_cols])
    
    # Create input form
    st.subheader("Input Features")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    # Dictionary to store input values
    input_data = {}
    
    # Create input fields for each feature
    for i, feature in enumerate(required_features):
        # Alternate between columns
        with col1 if i % 2 == 0 else col2:
            input_data[feature] = st.number_input(
                f"{feature}",
                value=0.0,
                format="%.2f",
                help=f"Enter value for {feature}"
            )
    
    # Add predict button
    if st.button("Predict"):
        try:
            # Create DataFrame from input
            input_df = pd.DataFrame([input_data])
            
            # Scale numeric features
            input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
            
            # Select features in correct order
            features = input_df[required_features].values.reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(features)
            
            # Display prediction
            st.success("Prediction made successfully!")
            st.subheader("Prediction Result")
            st.write(f"The predicted value is: {prediction[0]:.2f}")
            
            # Add visualization if needed
            st.subheader("Feature Values")
            st.bar_chart(input_data)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
else:
    st.error("Could not load the model. Please ensure 'best_gbr_model.pkl' is in the same directory.")

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit • GBR Model Predictor") 