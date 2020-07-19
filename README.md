# Household_Amenity_Detection
#### Application to detect 30 household amenities utilizing Detectron2 and Streamlit with mAP of 39%.

### Examples:
<p align="center"> <img src=https://github.com/jsantoso2/Household_Amenity_Detection/blob/master/Screenshot/demo-1.JPG width="700"></p>
<p align="center">Demo for Images<p align="center">

### Dataset
- Open Images v6: https://storage.googleapis.com/openimages/web/index.html

### Tools/Framework Used
- ML model: Detectron2 Model (https://github.com/facebookresearch/detectron2)
- Evaluation of ML models & Experiment Tracking: Weights and Biases (https://github.com/Cartucho/mAP)
- Web Application: Streamlit (https://www.streamlit.io/)
- Deployment: Google Cloud Platform App Engine
- Data Cleaning/Manipulation: Python 

### Procedure
- Download open images pictures into local file using https://github.com/EscVM/OIDv4_ToolKit
- Convert labels to detectron2 style labels
  - {'file_name': 'C:\\Users\\Jonathan Santoso\\workspace3\\Personal Projects\\Amenity detection\\dataset\\val\\007f71665b0812a7.jpg',
     'height': 768,
     'width': 1024,
     'image_id': '007f71665b0812a7',
     'annotations': [{'bbox': [0.0, 0.5952, 1024.0, 768.0],
     'bbox_mode': 0,
     'category_id': 24}]}
- Fine Tune Detectron2 Model for 20k iterations with evaluation for every 2k iterations
- Pick best model using validation mAP
- Create streamlit web application
- Output final images/videos
- Deploy to Google Cloud App Engine

### Results
- Detectron2 Fine-Tuning
<p align="center"> <img src=https://github.com/jsantoso2/Household_Amenity_Detection/blob/master/Screenshot/results.JPG height="450"></p>
<p align="center">Validation mAP, Average Precision<p align="center">
<p align="center"> <img src=https://github.com/jsantoso2/Household_Amenity_Detection/blob/master/Screenshot/results2.JPG height="450"></p>
<p align="center">Average Precision by Class<p align="center">

- Validation mAP  
  | Model | Training Time | mAP(%) | AP50% |  AP75% |
  |-------|-------------- |--------|-------|--------|
  | R50-FPN-3x | 2.5 hours | 29.035 | 47.212 | 29.823 |
  | R101-FPN-3x | 3 hours | 28.943 | 47.169 | 30.348 |
  | RN-50-3x | 6 hours | 39.412 | 58.921 | 42.521 |
  | RN-101-3x | 12 hours | 38.994 | 58.644| 41.618 |

- Test mAP  
  | Model | Training Time | mAP(%) | AP50% |  AP75% |
  |-------|-------------- |--------|-------|--------|
  | R50-FPN-3x | 2.5 hours | 30.18 | 48.59 | 29.82 |
  | R101-FPN-3x | 3 hours | 28.77 | 46.62 | 31.42 |
  | RN-50-3x | 6 hours | 38.49 | 58.92 | 45.52 |
  | RN-101-3x | 12 hours | 38.99 | 58.64 | 41.62 |
  
### Streamlit Web Application Screenshots
<p align="center"> <img src=https://github.com/jsantoso2/Household_Amenity_Detection/blob/master/Screenshot/app-1.JPG height="600"></p>
<p align="center">App Interface 1<p align="center">
<p align="center"> <img src=https://github.com/jsantoso2/Household_Amenity_Detection/blob/master/Screenshot/app-2.JPG height="600"></p>
<p align="center">App Interface 2<p align="center">
<p align="center"> <img src=https://github.com/jsantoso2/Household_Amenity_Detection/blob/master/Screenshot/app-3.JPG height="600"></p>
<p align="center">App Interface 3<p align="center">

### To Deploy Application to Google Cloud Engine:
- Deploy ready-application located in docker_test folder
- To deploy application to Google Cloud Engine:
  - Ensure that gcloud sdk is installed in local file system (https://cloud.google.com/sdk/install)
  - To list of all projects: gcloud projects list
  - To look at current project: gcloud config get-value project
  - Change to desired project: gcloud config set project (projectID)
  - To Deploy: gcloud app deploy

### References/Inspirations:
- https://medium.com/airbnb-engineering/amenity-detection-and-beyond-new-frontiers-of-computer-vision-at-airbnb-144a4441b72e
- https://github.com/mrdbourke/airbnb-amenity-detection

### Final Notes:
- To see how application works, please see Instructions.mp4 video
- To see more technical details, please see notes.docs for all my detailed notes
