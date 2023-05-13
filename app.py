import streamlit as st
from streamlit_option_menu import option_menu
import os
import requests
from PIL import Image, ImageOps
from io import BytesIO
import utils as ut
import cv2
import numpy as np
from skimage.metrics import structural_similarity
import imutils
import time

routes = os.environ["ROUTE"]

response = requests.get(url='https://katonic.ai/favicon.ico')
im = Image.open(BytesIO(response.content))

st.set_page_config(
    page_title='Pan_Card_Tampering_detector', 
    page_icon = im, 
)

def draw_something_on_top_of_page_navigation():
    st.sidebar.image('katonic_logo.png')

draw_something_on_top_of_page_navigation()

with st.sidebar:
    selected = option_menu(
        menu_title = "Main Menu",
        options = ["About","App"]
    )

if selected == "About":
    st.write("""
        # Pan Card Tampering Detection App
        """)

    img1 = Image.open('pan_card_tampering.png')
    st.image(img1,use_column_width=False)

    st.write("##### Purpose") 
    st.write("""This app is to detect tampering of PAN card using computer vision. This App will help different organizations or any normal people in detecting whether the Id i.e. the PAN card provided to them by thier employees or customers or anyone is original or not.""")

    st.write("##### Scope")   
    st.write("""This app can be used in different organizations where customers or users need to provide any kind of id in order to get themselves verified. 
                      The organization can use this project to find out whether the ID is original or fake. 
                      Similarly this can be used for any type of ID like adhar, voter id, etc.""")
    st.write("---")
    

if selected == "App":
    st.write("##### Let's check whether the PAN card of a customer is Fradulent or not")

    st.write("---")
    st.write("""
    ### User Guide

    ##### **There are two ways by which you can use this app.**
    
    **1. Type 1: Provide a PAN image that seems to be fradulent:**
            It'll output the tampering/similairty score with respect to a Typical real-PAN image

    **2. Type 2: Provide both original and fraud PAN images:**
            It'll output the tampering/similairty score between the PAN'S you provided
    """)
    select=st.selectbox('Provide the details regarding App usage type',['None','Type 1','Type 2'],key=2)
    if select == 'Type 1':
        file = st.file_uploader("Choose a Fraud image....", type=["jpg","png"])
        if file:
            with st.spinner('Wait for it...'):
                time.sleep(1)
            st.success('Done!')
        if file is None:
            st.text("Please upload an image file")
        else:
            #read_file = file.read()
            image = Image.open(file)
            image = ImageOps.fit(image, (250, 160) , Image.ANTIALIAS)
            image = np.asarray(image)
            #tampered_img = cv2.imread(tampered)
            original = cv2.imread('tamperd_file/original.jpg')

            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            tampered_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            (score, diff) = structural_similarity(original_gray, tampered_gray, full=True)
            diff = (diff * 255).astype("uint8")

            thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            for c in cnts:
            # applying contours on image
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            if st.button("Lets see_PAN_difference"):
                org_img = Image.fromarray(original)
                st.write("###### original image with contour")
                st.image(org_img)
                tmp_img = Image.fromarray(image)
                st.write("###### tampered image with contour")
                st.image(tmp_img)
                diff_img = Image.fromarray(diff)
                st.write("###### difference image with black")
                st.image(diff_img)
                thresh_img = Image.fromarray(thresh)
                st.write("###### threshold image with black")
                st.image(thresh_img)
            if st.button("similarity_score"):
                st.write(score)
                st.write("###### Note: Lower the similarity_score lower is the Similarity between real & fake PAN's")
                
    if select == "Type 2":
        original_img = st.file_uploader("Choose an Original image....", type=["jpg","png"])
        fraud_img = st.file_uploader("Choose a Fraud image....", type=["jpg","png"])
        
        if original_img is None or fraud_img is None :
            st.text("Please upload an image file")
        else:
            org_img = Image.open(original_img)
            tmp_img = Image.open(fraud_img)
            org_img = ImageOps.fit(org_img, (250, 160) , Image.ANTIALIAS)
            tmp_img = ImageOps.fit(tmp_img, (250, 160) , Image.ANTIALIAS)
            org_img = np.asarray(org_img)
            tmp_img =  np.asarray(tmp_img)

            original_gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
            tampered_gray = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY)
            
            (score, diff) = structural_similarity(original_gray, tampered_gray, full=True)
            diff = (diff * 255).astype("uint8")

            thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            for c in cnts:
            # applying contours on image
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(org_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(tmp_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            if st.button("Lets see_PAN_difference"):
                org_img = Image.fromarray(org_img)
                st.write("###### original image with contour")
                st.image(org_img)
                tmp_img = Image.fromarray(tmp_img)
                st.write("###### tampered image with contour")
                st.image(tmp_img)
                diff_img = Image.fromarray(diff)
                st.write("###### difference image with black")
                st.image(diff_img)
                thresh_img = Image.fromarray(thresh)
                st.write("###### threshold image with black")
                st.image(thresh_img)

            if st.button("similarity_score"):
                st.sucess(score)
                st.write("###### Note: Lower the similarity_score lower is the Similarity between real & fake PAN's")
                
  
