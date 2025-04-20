### ✅ Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import streamlit as st # Import Streamlit
import os # To check file existence

# --- Configuration ---
MODEL_FILENAME = 'random_forest_pipeline.pkl'
REPORT_FILENAME = 'model_report.txt'

# --- ML Pipeline Function ---
# This function contains the training, evaluation, and saving logic.
# It should only be run when needed (e.g., model file doesn't exist or user requests retraining).
def train_evaluate_and_save(show_plots=False):
    st.write("--- Starting Model Training and Evaluation ---")
    progress_bar = st.progress(0, text="Loading Data...")

    # Step 2: Load Data
    california = fetch_california_housing(as_frame=True)
    df = california.frame
    progress_bar.progress(10, text="Data Loaded.")

    # Step 3: Explore (Optional display in Streamlit)
    if st.checkbox("Show Data Exploration Details (during training)?"):
        st.subheader("Data Head")
        st.write(df.head())
        st.subheader("Data Description")
        st.write(df.describe())
        st.subheader("Null Values")
        st.write(df.isnull().sum())
        # Plotting (can slow down training display)
        # fig1, ax1 = plt.subplots()
        # sns.histplot(df['MedHouseVal'], bins=30, kde=True, ax=ax1)
        # ax1.set_title("Target Distribution")
        # st.pyplot(fig1)

    # Step 4-5: Split Data
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    # Save feature names for later use in UI
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    progress_bar.progress(25, text="Data Split.")

    # Step 6: Build Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    # Step 7: Hyperparameter Tuning
    st.write("Starting GridSearchCV (this may take a few minutes)...")
    params = {
        # Reduce complexity for faster demo if needed
        'regressor__n_estimators': [100], # [100, 200]
        'regressor__max_depth': [None, 10], # [None, 10, 20]
        'regressor__min_samples_split': [5] # [2, 5]
    }
    # Add verbose=1 to see progress in console if running locally
    grid_search = GridSearchCV(pipeline, params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_
    progress_bar.progress(75, text="GridSearchCV Finished.")

    # Step 8: Evaluate
    y_pred_test = evaluate_pipeline_streamlit("Tuned Random Forest Pipeline", best_pipeline, X_test, y_test)

    # Step 9: Save Pipeline
    joblib.dump(best_pipeline, MODEL_FILENAME)
    st.success(f"Pipeline saved as '{MODEL_FILENAME}'")
    progress_bar.progress(90, text="Model Saved.")

    # Step 12: Save Report
    mse = mean_squared_error(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    with open(REPORT_FILENAME, "w") as f:
        f.write(f"Best model: Random Forest (within Pipeline)\n")
        f.write(f"MSE: {mse}\n")
        f.write(f"MAE: {mae}\n")
        f.write(f"R^2: {r2}\n")
    st.success(f"Report saved as '{REPORT_FILENAME}'")

    # Step 13: Feature Importance (Optional Display)
    if st.checkbox("Show Feature Importance Plot (during training)?") or show_plots:
        try:
            importances = best_pipeline.named_steps['regressor'].feature_importances_
            fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
            ax_imp.barh(feature_names, importances)
            ax_imp.set_xlabel('Importance')
            ax_imp.set_title('Feature Importances (Random Forest)')
            st.pyplot(fig_imp)
        except Exception as e:
            st.warning(f"Could not generate feature importance plot: {e}")

    # Plot Predicted vs Actual (Optional Display)
    if st.checkbox("Show Predicted vs Actual Plot (during training)?") or show_plots:
         fig_pred, ax_pred = plt.subplots(figsize=(8, 6))
         ax_pred.scatter(y_test, y_pred_test, alpha=0.5)
         ax_pred.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Add diagonal line
         ax_pred.set_xlabel('True Values ($100k)')
         ax_pred.set_ylabel('Predicted Values ($100k)')
         ax_pred.set_title('Predictions vs Actual')
         ax_pred.grid(True)
         st.pyplot(fig_pred)

    progress_bar.progress(100, text="Training Complete.")
    st.balloons()
    return feature_names # Return feature names

# --- Helper Function for Evaluation Output in Streamlit ---
def evaluate_pipeline_streamlit(name, pipeline_model, X_t, y_t):
    y_pred = pipeline_model.predict(X_t)
    mse = mean_squared_error(y_t, y_pred)
    mae = mean_absolute_error(y_t, y_pred)
    r2 = r2_score(y_t, y_pred)
    st.subheader(f"{name} Evaluation Metrics:")
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2:.4f}")
    col2.metric("Mean Absolute Error", f"{mae:.4f}")
    col3.metric("Mean Squared Error", f"{mse:.4f}")
    return y_pred

# --- Model Loading Function (Cached) ---
# Use caching to avoid reloading the model on every interaction
@st.cache_resource
def load_model(filename=MODEL_FILENAME):
    if os.path.exists(filename):
        try:
            pipeline = joblib.load(filename)
            print("Model loaded from cache or file.") # Add print statement
            return pipeline
        except Exception as e:
            st.error(f"Error loading model '{filename}': {e}")
            return None
    else:
        print(f"Model file '{filename}' not found for loading.")
        return None

# --- Function to Get Feature Names ---
# Tries to get from model, falls back to hardcoded defaults if needed
def get_feature_names(pipeline):
    if pipeline and 'scaler' in pipeline.named_steps:
         try:
            # Attempt to get feature names from the scaler step if available
            f_names = pipeline.named_steps['scaler'].feature_names_in_
            if f_names is not None:
                return list(f_names)
         except AttributeError:
            pass # Scaler might not have feature_names_in_ attribute

    # Fallback: Default feature names for California Housing
    st.warning("Could not determine feature names from pipeline. Using default names.")
    return ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

# --- Streamlit App UI ---
st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("🏡 California House Price Prediction")
st.write("Using a Random Forest Regressor model trained via Scikit-learn.")

# Try to load the model
pipeline_model = load_model()

# Sidebar for options
with st.sidebar:
    st.header("Options")
    # Button to trigger retraining
    if st.button("Train / Retrain Model"):
        # Clear the cache before retraining to ensure the new model is loaded next time
        load_model.clear()
        # Call the training function
        with st.spinner("Training in progress... please wait."):
            trained_feature_names = train_evaluate_and_save(show_plots=True)
        # Attempt to reload the model immediately after training
        pipeline_model = load_model()
        # Optional: Use st.rerun() to refresh the whole app state after training if needed

    st.markdown("---")
    if os.path.exists(REPORT_FILENAME):
        st.header("Latest Model Report")
        try:
            with open(REPORT_FILENAME, "r") as f:
                st.text(f.read())
        except Exception as e:
            st.error(f"Could not read report file: {e}")
    else:
        st.info("Run training to generate the model report.")

# --- Main Prediction Interface ---
if pipeline_model is not None:
    st.header("Make a Prediction")
    st.write("Adjust the feature values below and click predict.")

    feature_names = get_feature_names(pipeline_model)

    # Use columns for better layout of inputs
    col_defs = st.columns(len(feature_names))
    input_values = []

    # Create number inputs for each feature dynamically
    with st.form("prediction_form"):
        input_features = {}
        cols = st.columns(4) # Adjust number of columns for input layout
        for i, feature in enumerate(feature_names):
            # Use columns to arrange inputs neatly
            with cols[i % 4]:
                 # Add reasonable defaults or min/max if known (using 0 as placeholder)
                input_features[feature] = st.number_input(
                    label=feature,
                    key=feature, # Unique key for each input
                    value=0.0,   # Default value
                    format="%.4f" # Format for float display
                )

        # Submit button for the form
        submitted = st.form_submit_button("Predict House Price")

        if submitted:
            # Get values in the correct order
            input_data_list = [input_features[fname] for fname in feature_names]
            input_data_array = np.array(input_data_list).reshape(1, -1)

            try:
                prediction = pipeline_model.predict(input_data_array)
                # Display prediction (Result is often in $100,000s for this dataset)
                predicted_value = prediction[0] * 100000
                st.success(f"Predicted Median House Value: ${predicted_value:,.2f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

else:
    st.warning("Model not found. Please train the model using the button in the sidebar.")
    # Provide default feature names even if model isn't loaded, for display consistency
    feature_names = get_feature_names(None)
    st.subheader("Expected Input Features:")
    st.write(", ".join(feature_names))

# --- Optional: Display Feature Importances from loaded model ---
st.markdown("---")
st.header("Model Insights (if available)")
if pipeline_model is not None and st.checkbox("Show Feature Importance from Loaded Model"):
     try:
         # Ensure feature names are consistent
         feature_names_for_plot = get_feature_names(pipeline_model)
         importances = pipeline_model.named_steps['regressor'].feature_importances_
         fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
         ax_imp.barh(feature_names_for_plot, importances)
         ax_imp.set_xlabel('Importance')
         ax_imp.set_title('Feature Importances (Random Forest)')
         st.pyplot(fig_imp)
     except Exception as e:
         st.warning(f"Could not generate feature importance plot from loaded model: {e}")