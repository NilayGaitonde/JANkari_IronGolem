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
from app import load_detector, load_classifier


def main():
    st.set_page_config(
        page_title="Savannah", layout="wide", initial_sidebar_state="auto"
    )
    st.title("Savannah")

    option = st.selectbox(
        "Select an option:", ("SenView", "Try a Demo (Lion)", "Try a Demo (Deer)")
    )

    classifier = load_classifier()
    detector = load_detector()

    # configuring what happens after selecting an option
    if option == "SenView":
        image = st.camera_input("Capture image")
        
        if image is not None:
            image = cv2.imdecode(np.frombuffer(image.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
            with st.spinner("Analyzing the image"):
                label = analyze_the_taken_image(image, classifier, detector)
            
            print(label)
            prediction_write_up = class_label_to_UI(label,Savannah)
            
            st.write(prediction_write_up)

    elif option == "Try a Demo (Lion)":
        image = cv2.imread("./assets/lion.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        st.write("Demo image: Lion")
        st.image(image)

        with st.spinner("Analyzing the image"):
            label = analyze_the_taken_image(image, classifier, detector)
        print(label)
        
        prediction_write_up = class_label_to_UI(label,Savannah)
            
        st.write(prediction_write_up)

    elif option == "Try a Demo (Deer)":
        image = cv2.imread("./assets/deer.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.write("Demo image: Deer")
        st.image(image)
        
        with st.spinner("Analyzing the image"):
            label = analyze_the_taken_image(image, classifier, detector)
        print(label)
        prediction_write_up = class_label_to_UI(label,Savannah)
            
        st.write(prediction_write_up)

    else:
        st.write("Select an option")

if __name__ == "__main__":
    main()