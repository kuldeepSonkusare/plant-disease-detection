import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
import tensorflow as tf
import tensorflow_hub as hub
import os
import numpy as np
from PIL import Image
from pathlib import Path
import webbrowser


# Load the model
model = hub.load("./model")

class_names = [
     "Apple___Apple_scab",
     "Apple___Black_rot",
     "Apple___Cedar_apple_rust",
     "Apple___healthy",
     "Blueberry___healthy",
     "Cherry_(including_sour)___Powdery_mildew",
     "Cherry_(including_sour)___healthy",
     "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
     "Corn_(maize)___Common_rust_",
     "Corn_(maize)___Northern_Leaf_Blight",
     "Corn_(maize)___healthy",
     "Grape___Black_rot",
     "Grape___Esca_(Black_Measles)",
     "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
     "Grape___healthy",
     "Orange___Haunglongbing_(Citrus_greening)",
     "Peach___Bacterial_spot",
     "Peach___healthy",
     "Pepper_bell___Bacterial_spot",
     "Pepper_bell___healthy",
     "Potato___Early_blight",
     "Potato___Late_blight",
     "Potato___healthy",
     "Raspberry___healthy",
     "Soybean___healthy",
     "Squash___Powdery_mildew",
     "Strawberry___Leaf_scorch",
     "Strawberry___healthy",
     "Tomato___Bacterial_spot",
     "Tomato___Early_blight",
     "Tomato___Late_blight",
     "Tomato___Leaf_Mold",
     "Tomato___Septoria_leaf_spot",
     "Tomato___Spider_mites Two-spotted_spider_mite",
     "Tomato___Target_Spot",
     "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
     "Tomato___Tomato_mosaic_virus",
     "Tomato___healthy"
]

cure_list = [
  "https://plantix.net/en/library/plant-diseases/100006/apple-scab",
  "https://plantix.net/en/library/plant-diseases/100001/apple-root-and-collar-rot",
  "https://www.planetnatural.com/pest-problem-solver/plant-disease/cedar-apple-rust/",
  'https://plantix.net/en/library/crops/apple',
  'https://plantix.net/en/library/crops/blueberry',
  "https://plantix.net/en/library/plant-diseases/100002/powdery-mildew",
  'https://plantix.net/en/library/crops/cherry',
  " https://plantix.net/en/library/plant-diseases/100107/grey-leaf-spot-of-maize",
  "https://plantix.net/en/library/plant-diseases/100082/common-rust-of-maize",
  " https://plantix.net/en/library/plant-diseases/100065/northern-leaf-blight",
  "https://plantix.net/en/library/crops/maize",
  "https://plantix.net/en/library/plant-diseases/100350/black-rot-of-grape",
  "https://plantix.net/en/library/plant-diseases/100140/esca",
  "https://plantix.net/en/library/plant-diseases/100208/powdery-mildew-of-grape",
  "https://plantix.net/en/library/crops/grape",
  "https://plantix.net/en/library/plant-diseases/300021/citrus-greening-disease",
  "https://vikaspedia.in/agriculture/crop-production/technologies-for-ne-india/fruits/peach-cultivation",
  'https://plantix.net/en/library/crops/peach',
  "https://plantix.net/en/library/plant-diseases/300003/bacterial-spot-of-pepper",
  "https://plantix.net/en/library/crops/pepper",
  "https://plantix.net/en/library/plant-diseases/100321/early-blight",
  "https://plantix.net/en/library/plant-diseases/100040/potato-late-blight",
  "https://plantix.net/en/library/crops/potato",
  "https://www.starkbros.com/growing-guide/how-to-grow/berry-plants/raspberry-plants/harvesting",
  "https://plantix.net/en/library/crops/soybean",
  "https://plantix.net/en/library/plant-diseases/100002/powdery-mildew",
  "https://plantix.net/en/library/plant-diseases/100028/common-leaf-spot",
  "https://plantix.net/en/community/questions/strawberry/20190105/let-me-know-best-practices-for-strawberries",
  "https://plantix.net/en/library/plant-diseases/300050/bacterial-spot-and-speck-of-tomato",
  "https://plantix.net/en/library/plant-diseases/100321/early-blight",
  "https://plantix.net/en/library/plant-diseases/100046/tomato-late-blight",
  "https://plantix.net/en/library/plant-diseases/100257/leaf-mold-of-tomato",
  "https://plantix.net/en/library/plant-diseases/100152/septoria-leaf-spot",
  "https://plantix.net/en/library/plant-diseases/500004/spider-mites",
  "https://plantix.net/en/library/plant-diseases/300050/bacterial-spot-and-speck-of-tomato",
  "https://plantix.net/en/library/plant-diseases/200036/tomato-yellow-leaf-curl-virus",
  " https://plantix.net/en/library/plant-diseases/200037/tobacco-mosaic-virus",
  "https://plantix.net/en/library/crops/tomato"
]

# print(len(class_names))
# print(len(cure_list))

## Page Title
st.set_page_config(page_title="Plant Disease Detection")



def get_predictions(input_image):
    img = Image.open(input_image)
    img = img.resize((224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = tf.expand_dims(img, axis=0)
    
    # Make the prediction
    prediction = model(img)
    predicted_class = tf.argmax(prediction, axis=1)
    
    # Return the predicted disease
    res = predicted_class[0].numpy()
    return class_names[int(res)]


with st.sidebar:
    choose = option_menu("Main Menu", ["Home", "About", "Steps To Use", "Detection","Contact"],
                         icons=['house', 'nut', 'kanban','camera fill', 'menu-button-wide', 'mailbox'],
                         menu_icon="app-indicator", default_index=0,
    )

def get_st_button_a_tag(url_link, button_name):
    """
    generate html a tag
    :param url_link:
    :param button_name:
    :return:
    """
    return f'''
    <a href={url_link}><button style="
    fontWeight: 400;
    padding: 0.25rem 0.75rem;
    borderRadius: 0.25rem;
    margin: 0px;
    lineHeight: 1.6;
    width: auto;
    userSelect: none;
    backgroundColor: #FFFFFF;
    border: 1px solid rgba(49, 51, 63, 0.2);">{button_name}</button></a>'''

def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

if choose == "Home":
    st.title("Home")
    st.markdown("---")
    title_container = st.container()
    col1, col2 = st.columns([12, 8])
    image1 = './img/plant-grow.gif'
    image2 = Image.open('./img/logo.png')
    st.image(image2, use_column_width="auto")
    # with title_container:
    #     with col1:
    #         st.image(image1, use_column_width="auto")
    #     with col2:
    #         st.image(image2, use_column_width="auto")
    st.markdown("<h1 style='text-align: center; color: green;'>Plant Disease Detection and Pesticide Suggestion with AI</h1>", unsafe_allow_html=True)
    intro_markdown = read_markdown_file("./doc/home.md")
    st.markdown(intro_markdown, unsafe_allow_html=True)

elif choose == "Detection":
    st.title("Plant Disease Detection")
    st.markdown("---")
    ## Input Fields
    uploaded_file = st.file_uploader("Upload a Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img = img.resize((224, 224))
        st.image(img)

    if st.button("Get Predictions"):
        suggestion = get_predictions(input_image=uploaded_file)
        if(suggestion == 'no disease'):
            st.success(suggestion)
        else:
            if 'healthy' in suggestion:
                st.success(f"Leaf is healthy : {suggestion}")
            else:
                indexDisease = class_names.index(suggestion)
                dd = cure_list[indexDisease]
                st.error(f"Disease detected : {suggestion}")
                st.markdown(get_st_button_a_tag(dd, 'Check Out Cure'), unsafe_allow_html=True)
            #st.write(f"check out cure at [link]({dd})")

elif choose == "About":
    st.title("About Us")
    st.markdown("---")        
    title_container = st.container()
    col1, col2 = st.columns([8, 12])
    image = Image.open('./img/about.png')
    with title_container:
        with col1:
            st.image(image, use_column_width="auto")
        with col2:
            intro_markdown = read_markdown_file("./doc/about.md")
            st.markdown(intro_markdown, unsafe_allow_html=True)
    

elif choose == "Steps To Use":
    st.title("Steps To Use")
    st.markdown("---")
    st.image('./img/steps.png', use_column_width="auto")
    intro_markdown = read_markdown_file("./doc/steps.md")
    st.markdown(intro_markdown, unsafe_allow_html=True)

elif choose == "Contact":
    st.title("Contact Us")
    st.markdown("---")
    with st.form(key='columns_in_form2',clear_on_submit=True):
        Name=st.text_input(label='Please Enter Your Name') #Collect user feedback
        Email=st.text_input(label='Please Enter Email') #Collect user feedback
        Message=st.text_input(label='Please Enter Your Message') #Collect user feedback
        submitted = st.form_submit_button('Submit')
        if submitted:
            st.write('Thanks for your contacting us. We will respond to your questions or inquiries as soon as possible!')
    title_container = st.container()
    col1, col2, col3 = st.columns([4, 4, 4])
    with title_container:
        with col1:
            st.markdown("Contact", unsafe_allow_html=True)
            st.markdown("+919090909090", unsafe_allow_html=True)
        with col2:
            st.markdown("Email", unsafe_allow_html=True)
            st.markdown("contact@gmail.com", unsafe_allow_html=True)
        with col3:
            st.markdown("Website", unsafe_allow_html=True)
            st.markdown("www.plantdisease.com", unsafe_allow_html=True)
