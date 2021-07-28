import cv2
import streamlit as st
import numpy as np
from PIL import Image
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
model_selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation()
mp_face_mesh=mp.solutions.face_mesh

model_face_mesh=mp_face_mesh.FaceMesh()

st.title("OpenCV")
st.subheader("Image Operations")
st.write("various operations with opencv")
st.write("Created by vishwajeet jagtap")


add_selectbox = st.sidebar.selectbox(
    "What operations would u like to perform",
    ("About","Change color","image blend","Selfie segmentation","Face mesh")
)
if add_selectbox=="About":
    st.write("this app is for demo purpose")

if add_selectbox=="Change color":
 
 color_schemes=st.sidebar.radio("choose color",("B","G","R"))
 image= None
 image_file_path=st.sidebar.file_uploader("upload image")
 if image_file_path is not None:
    image=np.array(Image.open(image_file_path))
    st.sidebar.image(image)  
    if color_schemes=="B":
        st.write("Converting to blue")
        zeros=np.zeros(image.shape[:2],dtype="uint8")
        b,g,r=cv2.split(image)
        blue_image=cv2.merge([zeros,zeros,b])
        st.image(blue_image)
    if color_schemes=="G":
        st.write("Converting to green")
        zeros=np.zeros(image.shape[:2],dtype="uint8")
        b,g,r=cv2.split(image)
        blue_image=cv2.merge([zeros,g,zeros])
        st.image(blue_image)
    if color_schemes=="R":
        st.write("Converting to red")
        zeros=np.zeros(image.shape[:2],dtype="uint8")
        b,g,r=cv2.split(image)
        blue_image=cv2.merge([r,zeros,zeros])
        st.image(blue_image)

elif add_selectbox=="image blend":
    st.write("Image blending")
    image_file_path1=st.sidebar.file_uploader("upload image1")
    image_file_path2=st.sidebar.file_uploader("upload image2")
    if image_file_path1 and image_file_path2 is not None:
        image=None
        image=np.array(Image.open(image_file_path1))
        image2=np.array(Image.open(image_file_path2))
        image2=cv2.resize(image2,(image.shape[1],image.shape[0]))
        blended_image=cv2.addWeighted(image,0.8,image2,0.3,gamma=0.5)
        st.image(blended_image)

elif add_selectbox=="Selfie segmentation":
    image= None
    st.write("selfie segmentation")
    image_file_path=st.sidebar.file_uploader("upload image")
    image_file_path2=st.sidebar.file_uploader("background image")
    
    if image_file_path and image_file_path2  is not None:       
        image=np.array(Image.open(image_file_path))
        image2=np.array(Image.open(image_file_path2))
        image2=cv2.resize(image2,(image.shape[1],image.shape[0]))
        BG_COLOR = (192, 192, 192)
        MASK_COLOR = (255, 255, 255)
        image_height,image_width,_=image.shape
        results = model_selfie_segmentation.process(image)
        condition1 = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        fg_image = np.zeros(image.shape, dtype=np.uint8)
        fg_image[:] = MASK_COLOR
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        output_image = np.where(condition1, image, image2)
        st.image(output_image)
elif add_selectbox=="Face mesh":
    st.write("Face mesh")
    image_file_path=st.sidebar.file_uploader("upload image")
    if image_file_path is not None:
        image=(np.array(Image.open(image_file_path)))
        st.sidebar.image(image)
        results=model_face_mesh.process(image) 

        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image,face_landmarks)
        st.image(image)   
