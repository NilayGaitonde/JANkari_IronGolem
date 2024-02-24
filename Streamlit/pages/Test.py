import streamlit as st
import numpy as np
from PIL import Image
from app import process_file
from dictionary import Savannah,code_to_label
from keras.api._v2.keras.models import load_model
import base64
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import image_dataset_from_directory
import tensorflow_hub as hub
from PIL import Image, ImageOps, ImageColor, ImageDraw, ImageFont, ImageOps
import time

def get_cropped_img(image, box):
    image = Image.fromarray(np.uint8(image)).convert("RGB")
    im_width, im_height = image.size
    ymin, xmin, ymax, xmax = tuple(box)
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                 ymin * im_height, ymax * im_height)
    im = image.crop((left,top,right,bottom))
    print(image.size,im.size)
    return im

def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img

def process_file(uploaded_file):
    image = Image.open(uploaded_file)
    image = image.resize((256, 256))
    return np.array(image)

@st.cache_resource(ttl=3600)

def load_plant_disease_model():
    module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/mobilenet_v2/1"
    return hub.load(module_handle).signatures['default']

def run_detector(detector, path,verbose=True):
    img = Image.open(path)

    converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    start_time = time.time()
    result = detector(converted_img)
    end_time = time.time()

    result = {key:value.numpy() for key,value in result.items()}
    if verbose:
        print("Found %d objects." % len(result["detection_scores"]))
        print("Inference time: ", end_time-start_time)
        for classes, scores,boxes in zip(result['detection_class_entities'],result['detection_scores'],result['detection_boxes']):
            print(f"{classes},{round(scores,2)},{boxes}")
    img  = np.asarray(img)
    image_cropped = get_cropped_img(img,result['detection_boxes'][0])
    return np.array(image_cropped)

def process_image(model, image):
    """
    returns a string that needs to be written using the streamlit write function
    """
    confidences = model.predict(image[np.newaxis, ...])
    class_pred =  np.argmax(confidences)
    label = class_code_to_label[class_pred]

    prediction_write_up = ""
    prediction_write_up += f"**_{label_to_name[label]}_** predicted with a confidence of {np.max(confidences) * 100:.2f}%  \n"
    prediction_write_up += f"&nbsp;   \n"
    prediction_write_up += plant_care_tips_md[label]

    return prediction_write_up

def main():
    st.set_page_config(
        page_title="PaudhaYodha", 
        page_icon=":potted_plant:", 
        layout="wide", 
        initial_sidebar_state="auto"
    )
    st.title('Test')
    option = st.selectbox("Select an option:", ("Take a photo", "Upload an image","Try a Demo"))

    if option == "Take a photo":
        # Use webcam to capture image
        image = st.camera_input("Capture image")
        if image is not None:
            image = process_file(image)
            prediction_write_up = process_image(model, image)
            st.write(prediction_write_up)

    elif option == "Upload an image":
        # Allow user to upload image
        uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "png", "jpeg", "heic", "webp"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
            image = run_detector(detector=load_plant_disease_model(),path=uploaded_file)
            st.image(image, caption='Cropped Image.', use_column_width=True)


if __name__ == "__main__":
    main()