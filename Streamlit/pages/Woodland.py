import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from dictionary import Woodlands
from app import load_detector, load_classifier
from functions import class_label_to_UI, analyze_the_taken_image


def main():
    st.set_page_config(
        page_title="Woodland", layout="wide", initial_sidebar_state="auto"
    )
    st.title("Woodlands")

    classifier = load_classifier()
    detector = load_detector()

    option = st.selectbox(
        "Select an option:", ("SenView", "Try a Demo (Wolf)", "Try a Demo (Rabbit)")
    )
    # configuring what happens after selecting an option
    if option == "SenView":
        image = st.camera_input("Capture image")

        if image is not None:
            image = cv2.imdecode(
                np.frombuffer(image.read(), dtype=np.uint8), cv2.IMREAD_COLOR
            )

            with st.spinner("Analyzing the image"):
                label = analyze_the_taken_image(image, classifier, detector)

            print(label)
            prediction_write_up = class_label_to_UI(label, Woodlands)

            st.write(prediction_write_up)

    elif option == "Try a Demo (Wolf)":
        image = cv2.imread("./assets/wolf.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.write("Demo image: Wolf")
        st.image(image)

        with st.spinner("Analyzing the image"):
            label = analyze_the_taken_image(image, classifier, detector)

        print(label)
        prediction_write_up = class_label_to_UI(label, Woodlands)

        st.write(prediction_write_up)

    elif option == "Try a Demo (Rabbit)":
        image = cv2.imread("./assets/rabbit.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.write("Demo image: Rabbit")
        st.image(image)

        with st.spinner("Analyzing the image"):
            label = analyze_the_taken_image(image, classifier, detector)

        print(label)
        prediction_write_up = class_label_to_UI(label, Woodlands)

        st.write(prediction_write_up)

    else:
        st.write("Select an option")


if __name__ == "__main__":
    main()
