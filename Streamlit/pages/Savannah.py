import streamlit as st
from app import process_file
import base64

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

def main():

    st.set_page_config(
        page_title="Savannah", 
        layout="wide", 
        initial_sidebar_state="auto"
    )

    st.title('Savannah')
    
    option = st.selectbox("Select an option:", ("SenView","Try a Demo"))
    if option == 'SenView':
        image = st.camera_input("Capture image")
        if image is not None:
            image = process_file(image)
            # prediction_write_up = process_image(model, image)
            # st.write(prediction_write_up)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
    elif option == 'Try a Demo':
        st.write("Upload a picture of your plant and let SenView identify it for you. Once the plant is identified, SenView will detect if a lion or hyena is present in the image.")
        demo =  st.text_input('Enter True or false')
        if demo == 'True':
            autoplay_audio('./thunder.mp3')
        else:
            st.write('No audio file found')
        
    else:
        st.write("Please select an option")



if __name__ == "__main__":
    main()

