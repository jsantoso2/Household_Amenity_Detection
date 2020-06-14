# common imports
import pandas as pd
import numpy as np
import os
import streamlit as st
from PIL import Image
import time
import cv2
import torch, torchvision

# Some basic setup:
import detectron2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances
from detectron2.data.catalog import Metadata


@st.cache
def load_model():
    cfg = get_cfg()

    # add project-specific config
    cfg.merge_from_file(os.path.join(os.getcwd(), 'config') + '/config_' + 'RN-101-3x' + '.yaml')

    # change inference type to cpu
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # load_model from saved file
    cfg.DATASETS.TEST = ("data_test", )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # set threshold for this model
    cfg.MODEL.WEIGHTS = os.path.join(os.getcwd(), 'output') + '/model_' + 'RN-101-3x' + '_final.pth'

    return DefaultPredictor(cfg)


@st.cache
def inference(im, predictor, data_metadata, threshold):
    # Convert PIL image to array
    im = np.asarray(im)
    
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], 
           metadata = data_metadata,
           scale = 0.5)

    # take only predictions with 25% confidence only for RetinaNet as they tend to overdraw
    filtered = outputs['instances'].to("cpu")._fields
    filtered_idx = []
    for i in range(len(filtered['scores'])):
        if filtered['scores'][i] >= threshold:
            filtered_idx.append(i)

    filt_instance = Instances(image_size=(im.shape[0],im.shape[1]), pred_boxes = outputs['instances']._fields['pred_boxes'][filtered_idx], 
              pred_classes = outputs['instances']._fields['pred_classes'][filtered_idx], 
              scores = outputs['instances']._fields['scores'][filtered_idx])

    v = v.draw_instance_predictions(filt_instance.to("cpu"))

    return v.get_image(), filt_instance


def main():
    st.title('Household Amenity Detection Project 👁')
    st.write("This Project is inspired by [Airbnb's machine learning powered amenity detection](https://medium.com/airbnb-engineering/amenity-detection-and-beyond-new-frontiers-of-computer-vision-at-airbnb-144a4441b72e).")
    st.write("And also by [Daniel Bourke's Airbnb amenity detection replication](https://github.com/mrdbourke/airbnb-amenity-detection).")

    st.subheader('How does it work?')
    st.write("1. Upload an image in either JPG or PNG or JPEG format.")
    st.write("2. Pick a probability threshold to determine what object + boxes to render.") 
    st.write("   Only objects with higher than threshold probability will be rendered.")
    st.write("3. Click the Make Prediction Button to run the model.")
    st.image(Image.open('demo.jpg'), use_column_width = True)


    st.subheader('Input File')

    objects = ['Bathtub', 'Bed', 'Billiard table', 'Ceiling fan', \
               'Coffeemaker', 'Couch', 'Countertop', 'Dishwasher', \
               'Fireplace', 'Fountain', 'Gas stove', 'Jacuzzi', \
               'Kitchen & dining room table', 'Microwave oven', \
               'Mirror', 'Oven', 'Pillow', 'Porch', 'Refrigerator',  \
               'Shower', 'Sink', 'Sofa bed', 'Stairs', 'Swimming pool', \
               'Television', 'Toilet', 'Towel', 'Tree house', 'Washing machine', 'Wine rack']

    # load model
    predictor = load_model()

    # create metadata
    data_metadata = Metadata(name = 'data_train', evaluator_type='coco', 
                         thing_classes = objects)


    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg", "JPG", "PNG", "JPEG"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make sure image is RGB
        image = image.convert("RGB")

        st.subheader('Output:')
        st.write("Pick a prediction threshold where only objects with probabilities above the threshold will be displayed!")
        pred_threshold = st.slider('Prediction Threshold:', 0.0, 1.0, 0.25)

        # get inference on image and display if button is clicked
        if st.button("Make Prediction"):
            start_time = time.time()

            # Some number in the range 0-1 (probabilities)
            with st.spinner("Doing Prediction..."):
                custom_pred, filt_instance = inference(image, predictor, data_metadata, pred_threshold)

            end_time = time.time()

            st.subheader('Predictions: ')
            # need to convert CV2 format to PIL format
            custom_pred = cv2.cvtColor(custom_pred, cv2.COLOR_RGB2BGR)
            st.image(custom_pred, caption = 'Predictions Image', use_column_width = True)

            st.write('Predicted Classes and Probabilities: ')
            # save predictions to dataframe
            pred_df = pd.DataFrame()
            object_name = []
            for elem in filt_instance.pred_classes.numpy():
                object_name.append(objects[elem])

            pred_df['Classes'] = object_name
            pred_df['Probabilities'] = filt_instance.scores.numpy()
            
            if pred_df.shape[0] == 0:
                st.write('No Objects Detected!')
            else:
                st.write(pred_df)

            # write prediction time
            pred_time = end_time - start_time
            st.write('Prediction Time: ' + ' {0:.2f}'.format(pred_time) + ' seconds')


    st.write("")
    st.subheader("What is under the hood?")
    st.write("Detectron2 RetinaNet model (PyTorch) and Streamlit web application")
    st.image(Image.open('logo.jpg'), use_column_width = True)

    st.subheader("Supported Classes/Objects:")
    st.write("• Bathtub          • Bed                 • Billiard Table")
    st.write("• Ceiling Fan      • Coffeemaker         • Couch")
    st.write("• Countertop       • Dishwasher          • Fireplace")
    st.write("• Fountain         • Gas Stove           • Jacuzzi")
    st.write("• Dining Table     • Microwave Oven      • Mirror")
    st.write("• Oven             • Pillow              • Porch")
    st.write("• Refrigerator     • Shower              • Sink")
    st.write("• Sofa bed         • Stairs              • Swimming Pool")
    st.write("• Television       • Toilet              • Towel")
    st.write("• Tree house       • Washing Machine     • Wine Rack")


if __name__ == '__main__':
    main()