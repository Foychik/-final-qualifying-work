import streamlit as st
import os
import shutil
from scripts.utils import run_bytetrack, run_deepsort


# === Интерфейс ===
st.set_page_config(page_title="Разработка Web-интерфейса для захвата и споровождения воздушной цели", layout="centered")
st.title("Захват и сопровождение воздушной цели")

# === Выбор модели ===
model_options = ["models/best_8m_LD.pt"]
model_path = st.selectbox("Выберите модель:", model_options)

# === Загрузка своей модели ===
custom_model = st.file_uploader("Или загрузите YOLO модель (.pt):", type=["pt"])
if custom_model:
    custom_model_path = os.path.join("models", custom_model.name)
    with open(custom_model_path, "wb") as f:
        f.write(custom_model.read())
    model_path = custom_model_path
    st.success(f"Модель {custom_model.name} загружена!")

# === Загрузка видео ===
st.subheader("Загрузка видео")
video_file = st.file_uploader("Перетащите видео или выберите файл:", type=["mp4", "mov", "avi", "gif"])

# === Выбор трекера ===
tracker = st.radio("Выберите метод трекинга:", ["ByteTrack", "DeepSORT"])

# === Кнопка запуска ===
if st.button("Запустить трекинг"):
    if video_file is not None:
        input_dir = "data/input"
        os.makedirs(input_dir, exist_ok=True)
        input_path = os.path.join(input_dir, video_file.name)

        with open(input_path, "wb") as f:
            f.write(video_file.read())

        st.info("Видео загружено. Выполняется обработка...")

        base_name = os.path.splitext(video_file.name)[0]

        if tracker == "ByteTrack":
            run_bytetrack(
                video_path=input_path,
                model_path=model_path,
                config="/bytetrack.yaml",
                output_dir="data/output/BT"
            )
            output_dir = "data/output/BT"
        else:
            run_deepsort(
                video_path=input_path,
                model_path=model_path,
                output_dir="data/output/DS"
            )
            output_dir = "data/output/DS"

        gif_path = os.path.join(output_dir, f"{base_name}.gif")
        video_path = os.path.join(output_dir, video_file.name)

        if os.path.exists(gif_path):
            st.image(gif_path, caption="Результат (GIF)", use_column_width=True)
            with open(gif_path, "rb") as f:
                st.download_button("Скачать GIF", f, file_name=os.path.basename(gif_path))
        elif os.path.exists(video_path):
            st.video(video_path)
            with open(video_path, "rb") as f:
                st.download_button("Скачать видео", f, file_name=os.path.basename(video_path))
        else:
            st.error("Не удалось найти файл результата.")
    else:
        st.warning("Пожалуйста, загрузите видео.")

# === О проекте ===
st.sidebar.title("О проекте")
st.sidebar.info(
    "Это веб-приложение позволяет загружать видео, запускать детекцию дронов и птиц с помощью YOLOv8 и отслеживать их с использованием ByteTrack или DeepSORT."
)
