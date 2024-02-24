import time
import numpy as np
from PIL import Image
import streamlit as st
from dictionary import biome
import base64

def process_file(uploaded_file):
    image = Image.open(uploaded_file)
    image = image.resize((256, 256))
    return np.array(image)

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
    Autorakshak is a  that helps you identify and protect endangered species from potential threats. It uses machine learning to identify the species and provide you with the best care tips. Just upload a picture of the species and let JanRakshak do the rest!
                """)
    
    # option = st.selectbox("Select an option:", ("Savannah", "Woodland","Desert","NA"))
    # if option == 'Savannah':
    #     st.write(biome[option])
    #     image = st.camera_input("Capture image")
    #     if image is not None:
    #         image = process_file(image)
    #         # prediction_write_up = process_image(model, image)
    #         # st.write(prediction_write_up)
    #         st.image(image, caption='Uploaded Image.', use_column_width=True)
    # elif option == 'Woodland':
    #     st.write(biome[option])
    #     image = st.camera_input("Capture image")
    #     if image is not None:
    #         image = process_file(image)
    #         # prediction_write_up = process_image(model, image)
    #         # st.write(prediction_write_up)
    #         st.image(image, caption='Uploaded Image.', use_column_width=True)
    # elif option == 'Desert':
    #     st.write(biome[option])
    #     image = st.camera_input("Capture image")
    #     if image is not None:
    #         image = process_file(image)
    #         # prediction_write_up = process_image(model, image)
    #         # st.write(prediction_write_up)
    #         st.image(image, caption='Uploaded Image.', use_column_width=True)
    # else:
    #     st.write("Please select an option")


    


if __name__ == "__main__":
    main()