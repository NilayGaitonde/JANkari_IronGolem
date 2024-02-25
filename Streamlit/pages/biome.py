import cv2
import numpy as np
import streamlit as st
from functions import (
    class_label_to_UI,
    analyze_the_taken_image,
    analyze_the_taken_image_wo_detector
)
from dictionary import animal_dict
from app import load_detector, load_classifier


def main():
    st.set_page_config(
        page_title="Savannah", layout="wide", initial_sidebar_state="auto"
    )
    st.title("Biome")

    option = st.selectbox(
        "Select an option:", ("SenView", "Upload a file", "Try a Demo (Deer)")
        # "Select an option:", ("SenView", "Upload a file")
    )
    
    classifier = load_classifier()
    detector = load_detector()

    # configuring what happens after selecting an option
    if option == "SenView":
        image = st.camera_input("Capture image")
        
        if image is not None:
            image = cv2.imdecode(np.frombuffer(image.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
            
            with st.spinner('Analyzing the image'):
                label = analyze_the_taken_image(image, classifier, detector)
            prediction_write_up = class_label_to_UI(label)
            
            st.write(prediction_write_up)

    elif option == "Upload a file":
        uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
            st.image(image)
            
            with st.spinner('Analyzing the image'):
                label = analyze_the_taken_image_wo_detector(image, classifier)
                
            prediction_write_up = class_label_to_UI(label, rule=animal_dict)
            print(prediction_write_up)
            st.write(prediction_write_up)

    elif option == "Try a Demo (Deer)":
        image = cv2.imread("./assets/dolphin.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.write("Demo image: Deer")
        st.image(image)
        
        with st.spinner('Analyzing the image'):
            label = analyze_the_taken_image_wo_detector(image, classifier)
        
        print(label)
        prediction_write_up = class_label_to_UI(label)
            
        st.write(prediction_write_up)

    else:
        st.write("Select an option")

if __name__ == "__main__":
    main()