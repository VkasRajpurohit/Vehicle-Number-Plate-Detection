import cv2

import numpy as np
import plotly.express as px
import pytesseract as pt

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from datetime import datetime

# Self written modules
from Logging import logger


class ObjectDetection:
    """It will perform the basic steps for object detection task & OCR.
       Written by : Vikram Singh
       Date: 07/12/2022"""

    def __init__(self):
        pass

    def image_processing(self, image_path, model):
        """Method Name: image_processing
           Description: It will return image & coordinates, after performing operations below -
           image read, image data processing, reshape, de-normalization & create bounding box."""
        logger.lg.info('image_processing - process started.')
        try:
            start_time = datetime.now()
            # Read Image
            image = load_img(image_path)  # PIL object
            image = np.array(image, dtype=np.uint8)  # 8 bit array (0, 255)
            image1 = load_img(image_path, target_size=(224, 224))
            logger.lg.info('Read Image: done.')

            # Data preprocessing
            image_arr_224 = img_to_array(image1) / 255.0  # Convert into array and get the normalized output
            h, w, d = image.shape  # Size of the original image
            logger.lg.info('Size of the original image - Height: {} & Width: {}'.format(h, w))
            logger.lg.info('Data preprocessing: done.')

            test_arr = image_arr_224.reshape(1, 224, 224, 3)
            logger.lg.info('Reshape operation: done.')
            # in order to pass this image of a model, we need to provide the data in the dynamic fourth dimension.
            # Here one indicates is a number of images, so we are just passing only one image.

            # Make predictions
            coords = model.predict(test_arr)

            # De-normalize the values
            denorm = np.array([w, w, h, h])
            coords = coords * denorm
            coords = coords.astype(np.int32)
            logger.lg.info('De-normalize operation: done.')

            # Draw bounding on top the image
            xmin, xmax, ymin, ymax = coords[0]
            p1, p2 = (xmin, ymin), (xmax, ymax)
            cv2.rectangle(image, p1, p2, (0, 255, 0), 3)
            logger.lg.info('Draw bounding on top the image: done.')
            end_time = datetime.now()
            execution_time = end_time - start_time
            logger.lg.info('Time taken for image_processing: {}.'.format(execution_time))

            return image, coords

        except Exception as e:
            logger.lg.warning('unable to complete request: {}'.format(e))

    def bounding_box(self, image):
        """Method Name: bounding_box
           Description: It will return Image with bounding box."""
        try:
            fig = px.imshow(image)
            fig.update_layout(width=750, height=500, margin=dict(l=10, r=10, b=10, t=10),
                              xaxis_title='Image with Bounding Box')
            logger.lg.info('Display Image with bounding box: done.')
            return fig
        except Exception as e:
            logger.lg.warning('unable to complete request: {}'.format(e))

    def ocr_with_pytesseract(self, image_path, coords):
        """Method Name: ocr_with_pytesseract
           Description: It will return the roi, figure of Bounding Box cropped image & Extracted text from the image."""
        try:
            # Bounding Box cropped image
            img = np.array(load_img(image_path))
            xmin, xmax, ymin, ymax = coords[0]
            roi = img[ymin:ymax, xmin:xmax]  # region of interest
            fig = px.imshow(roi)
            fig.update_layout(width=600, height=300,
                              margin=dict(l=10, r=10, b=10, t=10),
                              xaxis_title='Bounding Box cropped image')
            logger.lg.info('OCR - Bounding Box cropped image: done.')

            # Extract text from image
            text = pt.image_to_string(roi)
            logger.lg.info('Extract text from image: done.')
            logger.lg.info('Extracted Number plate: {}'.format(text))

            return roi, fig, text

        except Exception as e:
            logger.lg.warning('unable to complete request: {}'.format(e))




