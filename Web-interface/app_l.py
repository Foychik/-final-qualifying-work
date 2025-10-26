import streamlit as st
import os
from scripts.utils import run_bytetrack, run_deepsort

# === Настройки страницы ===
st.set_page_config(page_title="Захват и сопровождение целей", layout="centered")

# === Главный заголовок ===
st.markdown("<h1 style='text-align: center;'>Захват и сопровождение воздушной цели</h1>", unsafe_allow_html=True)
st.markdown("---")

# === Выбор модели ===
st.header("Выбор модели")

# Загрузка пользовательской модели
custom_model = st.file_uploader("Загрузите модель YOLOv8 (.pt):", type=["pt"])
if custom_model:
    os.makedirs("models", exist_ok=True)
    custom_model_path = os.path.join("models", custom_model.name)
    with open(custom_model_path, "wb") as f:
        f.write(custom_model.read())
    st.success(f"Модель {custom_model.name} загружена!")

# Выбор модели из списка
model_files = [f for f in os.listdir("models") if f.endswith(".pt")]
model_path = None
if model_files:
    selected_model = st.selectbox("Выберите модель из списка:", model_files)
    model_path = os.path.join("models", selected_model)
else:
    st.warning("Нет доступных моделей. Пожалуйста, загрузите модель.")

st.markdown("---")

# === Просмотр доступных видео ===
st.header("Просмотр доступных видео")

os.makedirs("data/input", exist_ok=True)
videos = [f for f in os.listdir("data/input") if f.endswith((".mp4", ".mov", ".avi", ".gif"))]
selected_video = None
if videos:
    selected_video = st.selectbox("Выберите видео для анализа:", videos)
    video_path = os.path.join("data/input", selected_video)
    if selected_video.endswith(".gif"):
        st.image(video_path)
    else:
        st.video(video_path)
else:
    st.info("Нет доступных видео. Загрузите файл ниже.")

# === Загрузка нового видео ===
st.header("📥 Загрузка нового видео")
new_video = st.file_uploader("Перетащите видео или выберите файл:", type=["mp4", "mov", "avi", "gif"])
if new_video:
    input_path = os.path.join("data/input", new_video.name)
    with open(input_path, "wb") as f:
        f.write(new_video.read())
    st.success(f"Видео {new_video.name} загружено! Перезапустите интерфейс для выбора.")

st.markdown("---")

# === Настройки трекинга ===
st.header(" Настройки трекинга")
tracker = st.radio("Выберите метод трекинга:", ["ByteTrack", "DeepSORT"])

# === Запуск обработки ===
if st.button("Запустить трекинг"):
    if not selected_video:
        st.warning("Сначала выберите видео из списка.")
    elif not model_path:
        st.warning("Сначала выберите или загрузите модель.")
    else:
        st.info("Обработка видео. Пожалуйста, подождите...")
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
                st.download_button("⬇️ Скачать GIF", f, file_name=os.path.basename(gif_path))
        elif os.path.exists(video_result_path):
            st.video(video_result_path)
            with open(video_result_path, "rb") as f:
                st.download_button("⬇️ Скачать видео", f, file_name=os.path.basename(video_result_path))
        else:
            st.error("Не удалось найти файл результата.")

# === О проекте ===
st.sidebar.title("📘 О проекте")
st.sidebar.info(
    "Это веб-приложение позволяет загружать видео, запускать детекцию дронов и птиц с помощью YOLOv8 "
    "и отслеживать их с использованием ByteTrack или DeepSORT. Разработка — ВКР 2025 Студента ПИ21-2 Васильченко Максим."
)
