import streamlit as st
import os
import shutil
from scripts.utils import run_bytetrack, run_deepsort

# === Интерфейс ===
st.set_page_config(page_title="Разработка Web-интерфейса для захвата и сопровождения воздушной цели", layout="centered")
st.title("Захват и сопровождение воздушной цели")

# === Загрузка пользовательской модели ===
custom_model = st.file_uploader("Загрузите модель YOLOv8 (.pt):", type=["pt"])
if custom_model:
    os.makedirs("models", exist_ok=True)
    custom_model_path = os.path.join("models", custom_model.name)
    with open(custom_model_path, "wb") as f:
        f.write(custom_model.read())
    st.success(f"Модель {custom_model.name} загружена!")

# === Выбор модели из директории ===
model_files = [f for f in os.listdir("models") if f.endswith(".pt")]
model_path = None
if model_files:
    selected_model = st.selectbox("Выберите модель из загруженных:", model_files)
    model_path = os.path.join("models", selected_model)
else:
    st.warning("Нет доступных моделей. Пожалуйста, загрузите модель.")

# === Просмотр доступных видео ===
st.subheader("Доступные видео")
os.makedirs("data/input", exist_ok=True)
videos = [f for f in os.listdir("data/input") if f.endswith((".mp4", ".mov", ".avi", ".gif"))]
selected_video = None
if videos:
    selected_video = st.selectbox("Выберите видео:", videos)
    video_path = os.path.join("data/input", selected_video)
    if selected_video.endswith(".gif"):
        st.image(video_path)
    else:
        st.video(video_path)
else:
    st.info("В папке 'data/input' нет видео. Загрузите файл ниже.")

# === Загрузка нового видео ===
st.subheader("Загрузка нового видео")
new_video = st.file_uploader("Перетащите видео или выберите файл:", type=["mp4", "mov", "avi", "gif"])
if new_video:
    input_path = os.path.join("data/input", new_video.name)
    with open(input_path, "wb") as f:
        f.write(new_video.read())
    st.success(f"Видео {new_video.name} загружено! Перезапустите для выбора.")

# === Выбор трекера ===
tracker = st.radio("Выберите метод трекинга:", ["ByteTrack", "DeepSORT"])

# === Кнопка запуска ===
if st.button(" Запустить трекинг"):
    if not selected_video:
        st.warning("Сначала выберите видео из списка выше.")
    elif not model_path:
        st.warning("Сначала выберите или загрузите модель.")
    else:
        st.info("Обработка видео...")
        base_name = os.path.splitext(selected_video)[0]

        if tracker == "ByteTrack":
            run_bytetrack(
                video_path=video_path,
                model_path=model_path,
                config="bytetrack.yaml",
                output_dir="data/output/BT"
            )
            output_dir = "data/output/BT"
        else:
            run_deepsort(
                video_path=video_path,
                model_path=model_path,
                output_dir="data/output/DS"
            )
            output_dir = "data/output/DS"

        gif_path = os.path.join(output_dir, f"{base_name}.gif")
        video_result_path = os.path.join(output_dir, f"{base_name}_{tracker.lower()}.mp4")

        if os.path.exists(gif_path):
            st.image(gif_path, caption="Результат (GIF)", use_column_width=True)
            with open(gif_path, "rb") as f:
                st.download_button("Скачать GIF", f, file_name=os.path.basename(gif_path))
        elif os.path.exists(video_result_path):
            st.video(video_result_path)
            with open(video_result_path, "rb") as f:
                st.download_button("Скачать видео", f, file_name=os.path.basename(video_result_path))
        else:
            st.error("Не удалось найти файл результата.")

# === О проекте ===
st.sidebar.title("О проекте")
st.sidebar.info(
    "Это веб-приложение позволяет загружать видео, запускать детекцию дронов и птиц с помощью YOLOv8 и отслеживать их с использованием ByteTrack или DeepSORT."
)
