import cv2
import base64
import numpy as np
import streamlit as st 
import tensorflow as tf
from dictionary import code_to_label, Savannah, Woodlands,animal_dict


def crop_resize_image(img):
    """
    this runs after the detection model has been run
    """
    #     print(img.shape)

    if img.shape[0] == img.shape[1]:
        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_LANCZOS4)

    elif img.shape[0] > img.shape[1]:
        new_width = int((256 / img.shape[0]) * img.shape[1])
        img = cv2.resize(img, dsize=(new_width, 256), interpolation=cv2.INTER_LANCZOS4)

        img = cv2.copyMakeBorder(
            img,
            0,
            0,
            abs(256 - new_width) // 2,
            abs(256 - new_width) // 2,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    else:
        new_height = int((256 / img.shape[1]) * img.shape[0])
        img = cv2.resize(img, dsize=(256, new_height), interpolation=cv2.INTER_LANCZOS4)

        img = cv2.copyMakeBorder(
            img,
            abs(256 - new_height) // 2,
            abs(256 - new_height) // 2,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    if img.shape[0] != 256:
        img = cv2.copyMakeBorder(
            img,
            abs(256 - img.shape[0]),
            0,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    if img.shape[1] != 256:
        img = cv2.copyMakeBorder(
            img,
            0,
            0,
            0,
            abs(256 - img.shape[1]),
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    #     print(img.shape)
    return img


def process_input_for_detection(image):
    img = image[np.newaxis, ...]
    img_tensor = tf.convert_to_tensor(img)
    return img, img_tensor


def process_image_based_on_detection(detection_result, image):
    assert type(image) == np.ndarray
    assert image.shape[0] == 1
    assert image.shape[3] == 3
    # also make sure image is a numpy array is in the shape (1, None, None, 3)

    box = detection_result["detection_boxes"].numpy()[0, 0]  # highest probability

    img_height = image.shape[2]
    img_width = image.shape[1]

# cropping the image using the bounding box
    image = image[
        0,
        int(box[0] * img_height) : int(box[2] * img_height),
        int(box[1] * img_width) : int(box[3] * img_width),
        :,
    ]

    cropped_resized_img = crop_resize_image(image)

    return cropped_resized_img[np.newaxis, ...]


def analyze_the_taken_image(
    image,
    classifier,
    detector,
):
    """
    this returns the class label of the image that was inputted into
    the streamlit UI.
    """
    img_np, img_tensor = process_input_for_detection(image)
    output_detector = detector(img_tensor)
    crop_img_np = process_image_based_on_detection(output_detector, img_np)

    
    confidences = classifier.predict(crop_img_np)
    class_pred = np.argmax(confidences)

    return code_to_label[class_pred]


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


def class_label_to_UI(label,rule ={}):
    if rule == animal_dict:
        if label in rule['predators']:
            st.write("Predator Alert")
            
            prediction_write_up = ""
            prediction_write_up += f"&nbsp;   \n"
            prediction_write_up += label
            audio_path = './assets/thunder.mp3'
            autoplay_audio(audio_path)
        else:
            prediction_write_up = ""
            prediction_write_up += f"&nbsp;   \n"
            prediction_write_up += "Area secured, you are rakshaked!"
    else:
        if label in rule:
            prediction_write_up = ""
            prediction_write_up += f"&nbsp;   \n"
            prediction_write_up += rule[label]
            audio_path = rule['audio']
            autoplay_audio(audio_path)
        
        else:
            prediction_write_up = ""
            prediction_write_up += f"&nbsp;   \n"
            prediction_write_up += "Area secured, you are rakshaked!"

        return prediction_write_up