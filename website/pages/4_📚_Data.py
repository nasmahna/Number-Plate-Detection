import streamlit as st
import os
from PIL import Image

#-----------------HEADER-----------------
st.title('License Plate Detection - Data')
#----------------------------------------

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

def tampilkan_gambar_dalam_folder(folder_path):
    # Mendapatkan daftar file dalam folder
    file_list = os.listdir(folder_path)

    # Filter hanya file gambar (misalnya, JPEG atau PNG)
    gambar_list = [file for file in file_list if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    if not gambar_list:
        st.write("Tidak ada gambar yang ditemukan dalam folder.")
        return

    # Menampilkan maksimal 5 gambar
    for i, gambar_file in enumerate(gambar_list[:5]):
        gambar_path = os.path.join(folder_path, gambar_file)

        # Menggunakan PIL untuk menampilkan gambar
        gambar_pil = Image.open(gambar_path)
        
        # Menampilkan gambar dengan judul
        st.image(gambar_pil, caption=f"Judul Gambar {i+1}: {gambar_file}", use_column_width=True)

    return gambar_pil  # Return the last image


path_k1 = '/Users/macbookair/Documents/Backup - PAAI/Dataset/K1 - Plat Hitam Motor '
path_k2 = '/Users/macbookair/Documents/Backup - PAAI/Dataset/K2 - Plat Putih Motor'
path_k3 = '/Users/macbookair/Documents/Backup - PAAI/Dataset/K3 - Plat Hitam Mobil'
path_k4 = '/Users/macbookair/Documents/Backup - PAAI/Dataset/K4 - Plat Putih Mobil'
path_k5 = '/Users/macbookair/Documents/Backup - PAAI/Dataset/K5 - Plat Istimewa'

# Pilih path yang ingin ditampilkan
selected_path = st.selectbox("Pilih Path:", [path_k1, path_k2, path_k3, path_k4, path_k5])

# Membuat wadah kosong untuk menampilkan gambar
container = st.empty()

# Tampilkan gambar untuk path yang dipilih
gambar_pil = tampilkan_gambar_dalam_folder(selected_path)

# Update konten wadah kosong saat ada perubahan
container.image(gambar_pil, caption=f"Judul Gambar 1: {selected_path}", use_column_width=True)



