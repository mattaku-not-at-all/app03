from ultralytics import YOLO
import streamlit as st
from PIL import Image
import io
import os

model = YOLO("yolov8x.pt")

uploaded_file = st.file_uploader("jpgファイルを選択してください", type='jpg')

# 作業用ディレクトリの設定
TMP_DIR = os.path.join(os.getcwd(), "tmp")
TMP_File = "tmp.jpg"
 
if not os.path.exists(TMP_DIR):
# ディレクトリが存在しない場合、ディレクトリを作成する
    os.makedirs(TMP_DIR)

if uploaded_file is not None:

    #オリジナルファイルの保存
    with open(os.path.join(TMP_DIR,TMP_File) , "wb") as f_org:
        f_org.write(uploaded_file.getvalue())
        img_org = Image.open(f_org.name)

    # YOLOモデルで物体検出を実行
    results = model(f_org.name, project="tmp", name="tmp", save=True)
    # 結果が保存されたディレクトリのパス
    saved_dir = results[0].save_dir

    # 完全なパスを作成
    saved_image_path = os.path.join(saved_dir, TMP_File)

    st.image(img_org, caption='orginal',use_column_width=True)
    st.image(saved_image_path, caption='物体検出結果', use_column_width=True)
