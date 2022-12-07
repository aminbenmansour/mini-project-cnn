import os
from pathlib import Path
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.utils import load_img, img_to_array

st.title('Tumor Classifier: Upload your X-Ray picture and find out whether the result is normal, benign or malignant')

base_dir = os.path.dirname(__file__)
cover_path = os.path.join(base_dir, 'xray_illustration.png')
image = Image.open(cover_path)
st.image(image, use_column_width=True)

model = tf.keras.models.load_model(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model'))

def predict_class(image):
	img = load_img(image, target_size=(128, 128), color_mode='grayscale')
	x = img_to_array(img)
	x = np.expand_dims(x, axis=0)
	images = np.vstack([x])
	images /= 255.0
	prediction = model.predict(images)

	return prediction


st.info('Upload DDSM scan for getting result ...')
file = st.file_uploader("Upload an image of an XRAY", type=["png"])



if file is None:
	# st.info('Upload DDSM scan for getting result ...')
	pass

else:
	target_path = os.path.join(base_dir, 'tempDir', file.name)

	with open(target_path, 'wb') as f:
		f.write(file.getbuffer())

	slot = st.empty()
	slot.text('Running inference....')
	target_image = Image.open(file)

	st.image(target_image, caption="Input Image", width = 400)
	pred = predict_class(target_path, model)
	class_names = ['CAN', 'BEN', 'NOR']
	result = class_names[np.argmax(pred)]
	output = 'The image is a ' + result
	slot.text('Done')
	st.success(output)

	os.remove(os.path.join(base_dir, 'tempDir', file.name))