import streamlit as st
import requests
import numpy as np
from streamlit_drawable_canvas import st_canvas

st.title("MNIST Digit Predictor")

st.write("Draw a digit (0-9):")
canvas = st_canvas(
    stroke_width=3,
    height=280,
    width=280,
    key="canvas",
)

if st.button("Make Prediction"):
    if canvas.image_data is not None:
        image = np.array(canvas.image_data)[:, :, 3]  
        image = image.reshape(-1).tolist()  
        response = requests.post("http://api:8000/predict", json={"image": image})
        prediction = response.json()["prediction"]
        st.write(f"Prediction: {prediction}")
    else:
        st.write("Please draw a digit before making a prediction.")
