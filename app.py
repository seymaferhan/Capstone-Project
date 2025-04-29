#GERMAN TRAFFIC SIGNS

import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


# Loading the model
model = load_model('gtsrb.h5')

def process_image(img): 
    img = img.resize((32, 32))
    img = np.array(img)
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)
    return img 

# Title and description
st.title(":vertical_traffic_light: German Traffic Signs :vertical_traffic_light: ")
st.write("Upload an Image and The Model Detect The Type of It")

# Image upload area
file = st.file_uploader('Select a Picture', type=['jpg', 'jpeg', 'png'])

if file is not None:
    img = Image.open(file)
    st.image(img, caption='Uploaded Image')
    
    # Process the image
    image = process_image(img)
    
    # Predict
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    # Class names 
    class_names = {0: 'Speed Limit (20 km/h)',
    1: 'Speed Limit (30 km/h)',
    2: 'Speed Limit (50 km/h)',
    3: 'Speed Limit (60 km/h)',
    4: 'Speed Limit (70 km/h)',
    5: 'Speed Limit (80 km/h)',
    6: 'End of Speed Limit (80 km/h)',
    7: 'Speed Limit (100 km/h)',
    8: 'Speed Limit (120 km/h)',
    9: 'No Passing',
    10: 'No Passing for Vehicles over 3.5 tons',
    11: 'Right of Way',
    12: 'Priority Road',
    13: 'Yield',
    14: 'Stop',
    15: 'No Entry',
    16: 'General Caution',
    17: 'Dangerous Curve to the Right',
    18: 'Dangerous Curve to the Left',
    19: 'Double Curve',
    20: 'Road Narrows on the Right',
    21: 'Road Narrows on the Left',
    22: 'Traffic Signals',
    23: 'Pedestrian Crossing',
    24: 'Children Crossing',
    25: 'Bicycles Crossing',
    26: 'Beware of Animals',
    27: 'End of All Restrictions',
    28: 'Turn Right Ahead',
    29: 'Turn Left Ahead',
    30: 'Ahead Only',
    31: 'Go Straight or Right',
    32: 'Go Straight or Left',
    33: 'Keep Right',
    34: 'Keep Left',
    35: 'Roundabout',
    36: 'End of Roundabout',
    37: 'No U-Turn',
    38: 'No Right Turn',
    39: 'No Left Turn',
    40: 'Speed Limit (40 km/h)',
    41: 'Speed Limit (50 km/h)',
    42: 'Speed Limit (60 km/h)',
    43: 'Speed Limit (70 km/h)'}
    
    # Show prediction result
    st.write(f"Predicted Class: **{class_names[predicted_class]}**")