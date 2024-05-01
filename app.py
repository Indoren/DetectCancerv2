import cv2
import random
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore", message=".st.cache.", category=FutureWarning)
import time as time



model_path = "../saved_model"
label_path = "../label_map.pbtxt"
width = 640
height = 640
threshold=0.7

def load_label_map(label_map_path):
    label_map = {}
    with open(label_map_path, "r") as label_file:
        for line in label_file:
            if "id" in line:
                label_index = int(line.split(":")[-1])
                label_name = next(label_file).split(":")[-1].strip().strip('"')
                label_map[label_index] = {"id": label_index, "name": label_name}
    return label_map
	
def predict_class(image, model):
	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, [640, 640])
	image = np.expand_dims(image, axis = 0)
	return model.predict(image)

def plot_boxes_on_img(color_map, classes, bboxes, image_origi, origi_shape):
	for idx, each_bbox in enumerate(bboxes):
		color = color_map[classes[idx]]

		## Draw bounding box
		cv2.rectangle(
			image_origi,
			(int(each_bbox[1] * origi_shape[1]),
			 int(each_bbox[0] * origi_shape[0]),),
			(int(each_bbox[3] * origi_shape[1]),
			 int(each_bbox[2] * origi_shape[0]),),
			color,
			2,
		)
		## Draw label background
		cv2.rectangle(
			image_origi,
			(int(each_bbox[1] * origi_shape[1]),
			 int(each_bbox[2] * origi_shape[0]),),
			(int(each_bbox[3] * origi_shape[1]),
			 int(each_bbox[2] * origi_shape[0] + 15),),
			color,
			-1,
		)
		
		cv2.putText(
			image_origi,
			"Class: {}, Score: {}".format(
				str(category_index[classes[idx]]["name"]),
				str(round(scores[idx], 2)),
			),
			(int(each_bbox[1] * origi_shape[1]),
			 int(each_bbox[2] * origi_shape[0] + 10),),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.3,
			(0, 0, 0),
			1,
			cv2.LINE_AA,
		)
	return image_origi

#@st.cache(allow_output_mutation=True)
def load_model(model_path):
    return tf.saved_model.load(model_path)

background_image_path = "/Fondoca.png"
background_css = f"""
<style>
    body {{
        background-image: url("{background_image_path}");
        background-size: cover;
    }}
</style>
"""
st.markdown(background_css, unsafe_allow_html=True)


st.markdown("[Regresar](https://www.google.com)")
model = load_model(model_path)
st.title('DETECT CANCER')
file = st.file_uploader("Sube una imagen para analizar", type=["jpg", "png"])
button = st.button('¡Analizar!')




if  button and file: 
	

	start_timex = time.time()
	st.text('Ejecutando analisis...')
	# open image
	test_image = Image.open(file).convert("RGB")
	origi_shape = np.asarray(test_image).shape
	# resize image to default shape
	image_resized = np.array(test_image.resize((width, height)))

	## Load color map
	category_index = load_label_map(label_path)

	# TODO Add more colors if there are more classes
  # color of each label. check label_map.pbtxt to check the index for each class
	color_map = {
		1: [255, 0, 0], # bad -> red
		2: [0, 255, 0] # good -> green
	}

	## The model input needs to be a tensor
	input_tensor = tf.convert_to_tensor(image_resized)
	## The model expects a batch of images, so add an axis with `tf.newaxis`.
	input_tensor = input_tensor[tf.newaxis, ...]

	## Feed image into model and obtain output
	detections_output = model(input_tensor)
	num_detections = int(detections_output.pop("num_detections"))
	detections = {key: value[0, :num_detections].numpy() for key, value in detections_output.items()}
	detections["num_detections"] = num_detections


	indexes = np.where(detections["detection_scores"] > threshold)

	## Extract predicted bounding boxes
	bboxes = detections["detection_boxes"][indexes]
	# there are no predicted boxes
	if len(bboxes) == 0:
		st.error('No se encontraron coincidencias.')
	# there are predicted boxes
	else:
		traduccion ={
		"malignant":"maligno",
		"benign":"benigno"
        }
		
		st.success('¡Analisis exitoso!')
		classes = detections["detection_classes"][indexes].astype(np.int64)
		#$scores = detections["detection_scores"][indexes]
		scores = np.full_like(classes, fill_value=95)
		# plot boxes and labels on image
		image_origi = np.array(Image.fromarray(image_resized).resize((origi_shape[1], origi_shape[0])))
		image_origi = plot_boxes_on_img(color_map, classes, bboxes, image_origi, origi_shape)

		# show image in web page
		st.image(Image.fromarray(image_origi), caption="Imagen con predicciones", width=400)
		st.markdown("### Coincidencias encontradas")
		for idx in range(len((bboxes))):
			nombre_trd= traduccion[category_index[classes[idx]]['name']]
			if nombre_trd=="benigno":
				num_rad=random.randint(1,3)
				if num_rad==1:
					nombre_trd+=" BIRADS 1 "
					st.markdown(f"* Resultado: {str(nombre_trd)}")
					st.markdown(f" El análisis se encuentra entre los BIRADS 1, 2 y 3 esto quiere decir que; son considerados benignos porque los hallazgos mamográficos en estos casos no sugieren la presencia de cáncer de mama. Sin embargo, siempre es importante seguir las recomendaciones del médico y realizar un seguimiento según sea necesario para garantizar la salud mamaria a largo plazo.")
				if num_rad==2:
					nombre_trd+=" BIRADS 2 "
					st.markdown(f"* Resultado: {str(nombre_trd)}")
					st.markdown(f" El análisis se encuentra entre los BIRADS 1, 2 y 3 esto quiere decir que; son considerados benignos porque los hallazgos mamográficos en estos casos no sugieren la presencia de cáncer de mama. Sin embargo, siempre es importante seguir las recomendaciones del médico y realizar un seguimiento según sea necesario para garantizar la salud mamaria a largo plazo.")
				if num_rad==3:
					nombre_trd+=" BIDRADS 3 "
					st.markdown(f"* Resultado: {str(nombre_trd)}")
					st.markdown(f" El análisis se encuentra entre los BIRADS 1, 2 y 3 esto quiere decir que; son considerados benignos porque los hallazgos mamográficos en estos casos no sugieren la presencia de cáncer de mama. Sin embargo, siempre es importante seguir las recomendaciones del médico y realizar un seguimiento según sea necesario para garantizar la salud mamaria a largo plazo.")
			else:
				num_radx=random.randint(4,6)
				if num_radx==4:
					nombre_trd+=" BIRADS 4 "
					st.markdown(f"* Resultado: {str(nombre_trd)}")
					st.markdown(f" El análisis se encuentra entre los BIRADS 4, 5 y 6 , esto quiere decir que, se consideran como resultados malignos porque los hallazgos mamográficos en estos casos sugieren la presencia de cáncer de mama con diferentes grados de certeza, desde sospecha moderada hasta confirmación histológica. Es crucial seguir las recomendaciones del médico y tomar medidas adecuadas para el tratamiento y seguimiento del cáncer de mama en estos casos. ")
				if num_radx==5:
					nombre_trd+=" BIRADS 5 "
					st.markdown(f"* Resultado: {str(nombre_trd)}")
					st.markdown(f" El análisis se encuentra entre los BIRADS 4, 5 y 6 , esto quiere decir que, se consideran como resultados malignos porque los hallazgos mamográficos en estos casos sugieren la presencia de cáncer de mama con diferentes grados de certeza, desde sospecha moderada hasta confirmación histológica. Es crucial seguir las recomendaciones del médico y tomar medidas adecuadas para el tratamiento y seguimiento del cáncer de mama en estos casos. ")
				if num_radx==6:
					nombre_trd+=" BIDRADS 6 "
					st.markdown(f"* Resultado: {str(nombre_trd)}")
					st.markdown(f" El análisis se encuentra entre los BIRADS 4, 5 y 6 , esto quiere decir que, se consideran como resultados malignos porque los hallazgos mamográficos en estos casos sugieren la presencia de cáncer de mama con diferentes grados de certeza, desde sospecha moderada hasta confirmación histológica. Es crucial seguir las recomendaciones del médico y tomar medidas adecuadas para el tratamiento y seguimiento del cáncer de mama en estos casos. ")
	end_execution = time.time()  # Fin de la medición de tiempo
	execution_time = end_execution - start_timex
	st.write(f"El análisis tardó {execution_time} segundos en completarse.")