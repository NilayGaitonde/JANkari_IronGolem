import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub

@st.cache_resource(ttl=3600)
def load_detector():
    return hub.load("./assets/detector_ssd_mobilenet")


@st.cache_resource(ttl=3600)
def load_classifier():
    return tf.keras.models.load_model("./assets/resnet_animal_v1.h5")

def main():
    st.set_page_config(
        page_title="JanRakshak", 
        page_icon=":potted_plant:", 
        layout="wide", 
        initial_sidebar_state="auto"
    )
    st.sidebar.success('test')

    st.title('AutoRakshak')

    st.subheader('An automated sentinal system to protect endangered species from potential threats')

    st.markdown("""
    Autorakshak is a  that helps you identify and protect endangered species from potential threats. It uses machine learning to identify the species and any potential threat it poses.Once it does so, It can detter the animal from the threat by using a combination of sound and light.
                """)

    load_classifier()
    load_detector()


if __name__ == "__main__":
    main()