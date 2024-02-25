import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from functions import (
    class_label_to_UI,
    analyze_the_taken_image
)
from dictionary import Savannah


@st.cache_resource(ttl=3600)
def load_detector():
    return hub.load("./assets/detector_ssd_mobilenet")


@st.cache_resource(ttl=3600)
def load_classifier():
    return tf.keras.models.load_model("./assets/resnet_animal_v1.h5")


def main():
    st.set_page_config(
        page_title="Savannah", layout="wide", initial_sidebar_state="auto"
    )
    st.title("Savannah")


    # TODO: Implement threading for these two
    detector = load_detector()
    classifier = load_classifier()

    option = st.selectbox(
        "Select an option:", ("SenView", "Try a Demo (Lion)", "Try a Demo (Deer)")
    )
    # configuring what happens after selecting an option
    if option == "SenView":
        image = st.camera_input("Capture image")
        
        if image is not None:
            image = cv2.imdecode(np.frombuffer(image.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
            label = analyze_the_taken_image(image, classifier, detector)
            prediction_write_up = class_label_to_UI(label,Savannah)
            
            st.write(prediction_write_up)

    elif option == "Try a Demo (Lion)":
        image = cv2.imread("./assets/lion.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        st.write("Demo image: Lion")
        st.image(image)

        label = analyze_the_taken_image(image, classifier, detector)
        prediction_write_up = class_label_to_UI(label,Savannah)
            
        st.write(prediction_write_up)

    elif option == "Try a Demo (Deer)":
        image = cv2.imread("./assets/deer.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.write("Demo image: Deer")
        st.image(image)
        
        label = analyze_the_taken_image(image, classifier, detector)
        prediction_write_up = class_label_to_UI(label,Savannah)
            
        st.write(prediction_write_up)

    else:
        st.write("Select an option")

if __name__ == "__main__":
    main()