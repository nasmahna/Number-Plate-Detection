import streamlit as st

#>>>>>>>>>>>STYLING<<<<<<<<<<<<<
# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)

#-----------------HEADER-----------------
st.sidebar.success('Select a page above')
st.title('Welcome👋,')
st.subheader('License Plate Detection App')

#---------------CONTENTS---------------
st.text('''
        Hello, this is a project by Nasmah Nur Amiroh. 
        This webpage provides you to perform various tasks related to licence plate detection, including training and testing, in addition to display the data used for training.
        YoloV8 and OCR were used in this research to detect licence plates.

        ''')
#----------------------------------------

