import streamlit as st 
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Tensorflow model prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('../trained model/plant_desies_trained_model.keras')
    image = Image.open(test_image).convert("RGB")
    image = image.resize((128, 128))
    input_arr = np.array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Single image to batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, width=700) 
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases.
    """)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset consists of about 87K RGB images of healthy and diseased crop leaves categorized into 38 classes.
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=['jpg','jpeg','png'])
    
    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, width=700)
        
        if st.button("Predict"):
            with st.spinner("Please Wait..."):
                result_index = model_prediction(test_image)
                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
                    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy'
                ]
                st.success(f"Model predicts it is: **{class_name[result_index]}**")
