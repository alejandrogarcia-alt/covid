import streamlit as st
import os
import subprocess
import sys
from PIL import Image
import numpy as np

# Add src directory to path to import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from predict import predict_and_visualize, preprocess_image
from data_preprocessing import get_class_names

# --- Page Configuration ---
st.set_page_config(
    page_title="COVID-19 Detection Assistant",
    page_icon="🩺",
    layout="wide"
)

# --- Constants ---
MODEL_PATH = 'covid_detection_model.h5'
HISTORY_DIR = 'training_history'
DEFAULT_IMAGE_PATH = 'default_test_image.png' # A default image to show

# --- Helper Functions ---
def run_training():
    """Runs the main training script and streams its output."""
    st.info("Starting the training process... This may take a while. Please see the console for real-time progress.")
    
    process = subprocess.Popen(
        [sys.executable, os.path.join('src', 'train.py')],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    output_placeholder = st.empty()
    output_text = ""
    for line in process.stdout:
        output_text += line
        output_placeholder.text_area("Training Log", output_text, height=400)
    
    process.wait()
    
    if process.returncode == 0:
        st.success("Training completed successfully!")
        st.balloons()
    else:
        st.error("An error occurred during training. Please check the log above.")

# --- Main App ---
st.title("🩺 AI Assistant for COVID-19 Detection from Chest X-Rays")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Inference", "Training Dashboard"])

# ==============================================================================
# --- Inference Page ---
# ==============================================================================
if page == "Inference":
    st.header("Test the Model with an X-Ray Image")

    st.write("""
        Upload a chest X-ray image to get a prediction from the trained model.
        The model will classify the image as **Normal** or **COVID-compatible**.
        A heatmap will be generated to show which parts of the image the model focused on for its decision.
    """)

    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location to use with our functions
        with open("temp_image.png", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(uploaded_file, caption="Uploaded X-Ray", use_column_width=True)
        
        with st.spinner("Analyzing the image..."):
            try:
                if not os.path.exists(MODEL_PATH):
                    st.error(f"Model file not found at '{MODEL_PATH}'. Please train the model first on the 'Training Dashboard' page.")
                else:
                    # Run prediction and visualization
                    predicted_class, confidence, superimposed_img = predict_and_visualize("temp_image.png", MODEL_PATH)
                    
                    # Display results
                    st.subheader("Analysis Result")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if predicted_class == "COVID":
                            st.error(f"Prediction: **{predicted_class}**")
                        else:
                            st.success(f"Prediction: **{predicted_class}**")
                        st.metric(label="Confidence", value=f"{confidence:.2%}")

                    with col2:
                        st.image(superimposed_img, caption="Model Attention Heatmap", use_column_width=True)

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")

# ==============================================================================
# --- Training Dashboard Page ---
# ==============================================================================
elif page == "Training Dashboard":
    st.header("Train and Evaluate the Model")

    st.write("""
        Here you can initiate the model training process. The process follows the two-phase strategy:
        1.  **Feature Extraction:** Trains only the newly added classification layers.
        2.  **Fine-Tuning:** Unfreezes some layers of the base model and continues training with a lower learning rate.
        
        The training logs will be displayed below in real-time.
    """)

    if st.button("🚀 Start Training"):
        run_training()

    st.subheader("Training Results")
    st.write("After training is complete, the performance graphs will be displayed here.")

    # Check if training history images exist and display them
    if os.path.exists(HISTORY_DIR):
        history_files = [f for f in os.listdir(HISTORY_DIR) if f.endswith('.png')]
        if history_files:
            for history_file in sorted(history_files):
                image = Image.open(os.path.join(HISTORY_DIR, history_file))
                st.image(image, caption=history_file.replace('_', ' ').title(), use_column_width=True)
        else:
            st.info("No training history found. Run the training to generate performance plots.")
    else:
        st.info("The 'training_history' directory does not exist. Run the training first.")
