import streamlit as st
import numpy as np
from PIL import Image
from app import process_file
from dictionary import Savannah,code_to_label
from keras.api._v2.keras.models import load_model
import base64

@st.cache_resource(ttl=3600)
def load_plant_disease_model():
    return load_model("./assets/resnet_animal_v1.h5")

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )

def process_image(model, image):
    """
    returns a string that needs to be written using the streamlit write function
    """
    image = image.resize((256, 256))
    image = np.array(image)
    confidences = model.predict(image[np.newaxis, ...])
    class_pred =  np.argmax(confidences)
    label = code_to_label[class_pred]
    
    if label in Savannah:
        
        
        prediction_write_up = ""
        # prediction_write_up += f"**_{label}_** predicted with a confidence of {np.max(confidences) * 100:.2f}%  \n"
        prediction_write_up += f"&nbsp;   \n"
        prediction_write_up += Savannah[label]
        autoplay_audio('./assets/thunder.mp3')
    else:
        prediction_write_up = ""
        # prediction_write_up += f"**_{label}_** predicted with a confidence of {np.max(confidences) * 100:.2f}%  \n"
        prediction_write_up += f"&nbsp;   \n"
        prediction_write_up += "Area secured, Keep moving forward!"
   

    return label,prediction_write_up

def main():

    st.set_page_config(
        page_title="Savannah", 
        layout="wide", 
        initial_sidebar_state="auto"
    )

    st.title('Savannah')
    
    model = load_plant_disease_model()

    option = st.selectbox("Select an option:", ("SenView","Try a Demo"))
    if option == 'SenView':
        image = st.camera_input("Capture image")
        if image is not None:
            image = Image.open(image)
            label,prediction_write_up = process_image(model, image)
            st.write(prediction_write_up)
            
    elif option == 'Try a Demo':
        st.write("Upload a picture of your plant and let SenView identify it for you. Once the plant is identified, SenView will detect if a lion or hyena is present in the image.")
        image = (Image.open("./assets/lion.jpg"))
        st.write("Demo image: Lion")
        st.image(image)
        
        # with st.spinner('loading prediction'):
        #     time.sleep(0.8)
        # st.write("#### Prediction:")
        label,prediction_write_up = process_image(model, image)
        st.write(prediction_write_up)


        
        
        
    else:
        st.write("Please select an option")



if __name__ == "__main__":
    main()

