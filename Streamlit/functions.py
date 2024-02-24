def crop_resize_image(img):
#     print(img.shape)
    
    if img.shape[0] == img.shape[1]:
        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_LANCZOS4)
    
    elif img.shape[0] > img.shape[1]:
        new_width = int((256/img.shape[0]) * img.shape[1])
        img = cv2.resize(img, dsize=(new_width, 256), interpolation=cv2.INTER_LANCZOS4)
        
        img = cv2.copyMakeBorder(
            img,
            0, 0,
            abs(256 - new_width) // 2, abs(256 - new_width) // 2,  
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        
    else:
        new_height = int((256/img.shape[1]) * img.shape[0])
        img = cv2.resize(img, dsize=(256, new_height), interpolation=cv2.INTER_LANCZOS4)
  
        img = cv2.copyMakeBorder(
            img, 
            abs(256 - new_height) // 2, abs(256 - new_height) // 2,  
            0, 0,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
    
    if img.shape[0] != 256:
        img = cv2.copyMakeBorder(
            img,
            abs(256 - img.shape[0]),
            0, 0, 0,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )

    if img.shape[1] != 256:
        img = cv2.copyMakeBorder(
            img,
            0, 0, 0,
            abs(256 - img.shape[1]),
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
    
#     print(img.shape)
    return img


def process_input_for_detection(image_path):
    img = plt.imread(image_path)[np.newaxis, ...]
    img_tensor = tf.convert_to_tensor(img)
    return img, img_tensor


def process_image_based_on_detection(detection_result, image):
    assert type(image) == np.ndarray
    assert image.shape[0] == 1
    assert image.shape[3] == 3
    # also make sure image is a numpy array is in the shape (1, None, None, 3)
    
    box = detection_result["detection_boxes"].numpy()[0, 0] # highest probability
    
    img_height = image.shape[2]
    img_width = image.shape[1]
    image = image[0, int(box[0]*img_height):int(box[2]*img_height), int(box[1]*img_width):int(box[3]*img_width), :]
    
    cropped_resized_img = crop_resize_image(image)
    
    return cropped_resized_img[np.newaxis, ...]