from ultralytics import YOLO
import streamlit as st
from PIL import Image
import io
import os
import uuid

model = YOLO("yolov8x.pt")

uploaded_file = st.file_uploader("jpgファイルを選択してください", type='jpg')

# 作業用親ディレクトリの設定
TMP_DIR_NAME = "tmp"
ORG_DIR_NAME = "original"
PRED_DIR_NAME = "pred"
TMP_DIR = os.path.join(os.getcwd(), TMP_DIR_NAME)
ORG_DIR = os.path.join(os.getcwd(), TMP_DIR_NAME, ORG_DIR_NAME)
PRED_DIR = os.path.join(os.getcwd(), TMP_DIR_NAME, PRED_DIR_NAME)

# 作業用親ディレクトリが存在しない場合、ディレクトリを作成する
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)
if not os.path.exists(ORG_DIR):
    os.makedirs(ORG_DIR)
if not os.path.exists(PRED_DIR):
    os.makedirs(PRED_DIR)
 
if uploaded_file is not None:   #アップロードされた場合の処理

    #アップロードしたファイル名
    TMP_File = "tmp.jpg"

    #一時フォルダ用UUIDの取得
    uuid = uuid.uuid4()
    UUID_ORG_DIR = os.path.join(ORG_DIR, str(uuid))
    UUID_PRED_DIR = os.path.join(PRED_DIR, str(uuid))

    if not os.path.exists(UUID_ORG_DIR):
    # ディレクトリが存在しない場合、ディレクトリを作成する
        os.makedirs(UUID_ORG_DIR)

    #オリジナルファイルの保存
    with open(os.path.join(UUID_ORG_DIR,TMP_File) , "wb") as f_org:
        f_org.write(uploaded_file.getvalue())
        img_org = Image.open(f_org.name)

    # YOLOモデルで物体検出を実行
    results = model(f_org.name, project=PRED_DIR, name=str(uuid), save=True)
    # 結果が保存されたディレクトリのパス
    saved_dir = results[0].save_dir

    # 完全なパスを作成
    saved_image_path = os.path.join(saved_dir, TMP_File)
    img_pred = Image.open(saved_image_path)

    #画像表示
    col1, col2 = st.columns(2)

    with col1:
        st.header("オリジナル画像")
        st.image(img_org, use_column_width=True)

    with col2:
        st.header("検知画像")
        st.image(img_pred, use_column_width=True)
