import os

import tensorflow as tf

from datetime import datetime
from flask import Flask, render_template, request
from flask_cors import cross_origin
from werkzeug.utils import secure_filename

# Self written modules
from Logging import logger
from Modules.ObjectDetection import ObjectDetection

app = Flask(__name__)  # app as object created
logger.lg.info('app start- working fine')

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, r'static/upload')
SAVED_MODEL = os.path.join(BASE_PATH, r'Saved_Model/model_number_plate.h5')
ObjectDetection = ObjectDetection()


@app.route('/', methods=['GET', 'POST'])  # To render Home_Page
@cross_origin()
def home_page():
    """landing to home_page"""
    logger.lg.info('landing on home page.')
    return render_template('index.html')


@app.route('/about')  # To render about page
@cross_origin()
def about():
    """about the app information"""
    logger.lg.info('about page.')
    return render_template('about.html')


@app.route('/contact')  # To render contact page
@cross_origin()
def contact():
    """contact information"""
    logger.lg.info('contact page.')
    return render_template('contact.html')


@app.route('/result', methods=['GET', 'POST'])  # To render result page
@cross_origin()
def result():
    """It will process the uploaded image by user & display result.
       Written by : Vikram Singh
       Date: 07/12/2022"""
    if request.method == 'POST':
        logger.lg.info('Request = POST.\n')
        process_start_time = datetime.now()
        logger.lg.info('Process started.')
        try:
            uploaded_file = request.files['image_name']
            if uploaded_file:
                filename = secure_filename(uploaded_file.filename)  # for security
                path_save = os.path.join(UPLOAD_PATH, filename)
                uploaded_file.save(path_save)
                logger.lg.info('File uploaded successfully.')

                # Loading saved model
                try:
                    start_time = datetime.now()
                    model = tf.keras.models.load_model(SAVED_MODEL)
                    logger.lg.info('Model loaded successfully.')
                    end_time = datetime.now()
                    model_load_time = end_time - start_time
                    logger.lg.info('Time taken for loading saved model: {}.'.format(model_load_time))
                except Exception as e:
                    logger.lg.warning('unable to complete request: {}'.format(e))

                image, cords = ObjectDetection.image_processing(path_save, model)
                roi, fig_img_cropped_bb, extracted_text = ObjectDetection.ocr_with_pytesseract(path_save, cords)

                logger.lg.info('Process completed successfully.')
                process_end_time = datetime.now()
                process_execution_time = process_end_time - process_start_time
                logger.lg.info('Time taken for the entire process: {}.\n'.format(process_execution_time))
            return render_template('index.html', upload=True, upload_image=filename, extracted_text=extracted_text)
        except Exception as e:
            logger.lg.warning('unable to complete request: {}'.format(e))


if __name__ == '__main__':  # on running python app.py
    app.run(debug=True)
