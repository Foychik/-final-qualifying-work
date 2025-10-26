from ultralytics import YOLO
from scripts.utils import run_deepsort

if __name__ == "__main__":
    run_deepsort(
        video_path="data/input",
        model_path="models/best_8m_LD.pt",
      #   tracker_name="deepsort",
        output_dir="data/output/DS"
    )
