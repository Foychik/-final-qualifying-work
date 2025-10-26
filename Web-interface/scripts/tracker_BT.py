from ultralytics import YOLO
from scripts.utils import run_bytetrack


if __name__ == "__main__":
    run_bytetrack(
        video_path="data/input",
        model_path="models/best_8m_LD.pt",
        config="./Web-interface/trackers-config/deepsort.yaml",
        output_dir="data/output/BT"
    )
