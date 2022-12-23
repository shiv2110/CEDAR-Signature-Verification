import streamlit as st
import numpy as np 
import matplotlib.pyplot as plt

from PIL import Image
import preprocess
import predict

st.set_page_config(page_title = 'Signature Forgery Identification', page_icon = '📝')

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)



padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)


st.title('Is the signature real or forged?')


uploaded_img = st.file_uploader('Upload a scanned image',
                               type = 'jpg', 
                               help = 'an image file with .jpg extension to be uploaded from your local FS')


if uploaded_img:
   
    st.image(uploaded_img, caption = 'signature', width = 500)
    test_image = Image.open(uploaded_img)
    test_image = np.array(test_image)

    if st.button('Get Result'):

        bin_img = preprocess.preprocess(test_image)
     
        result = predict.predict(bin_img)
        st.subheader('Result of Convolutional Siamese Neural Network')
        if result < 0.5:
            st.header('The signature is FORGED')
        else:
            st.header('The signature is genuine')

        # st.header(result)
