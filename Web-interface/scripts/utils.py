import os
import cv2
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from moviepy.editor import VideoFileClip
import random

def get_color(idx):
    random.seed(idx)
    return [random.randint(0, 255) for _ in range(3)]

def run_bytetrack(video_path, model_path, output_dir, config="bytetrack.yaml"):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(3)), int(cap.get(4))

    filename = os.path.splitext(os.path.basename(video_path))[0]
    video_out_path = os.path.join(output_dir, f"{filename}_bytetrack.mp4")
    gif_out_path = os.path.join(output_dir, f"{filename}.gif")
    os.makedirs(output_dir, exist_ok=True)
    out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    history = defaultdict(list)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(source=frame, tracker=f"./trackers-config/{config}", persist=True, conf = 0.6)[0]
        annotated = results.plot()

        for box in results.boxes:
            if box.id is not None:
                tid = int(box.id.item())
                center = tuple(map(int, box.xywh[0][:2]))
                history[tid].append(center)
                for i in range(1, len(history[tid])):
                    cv2.line(annotated, history[tid][i - 1], history[tid][i], get_color(tid), 2)

        out.write(annotated)

    cap.release()
    out.release()

    VideoFileClip(video_out_path).write_gif(gif_out_path, fps=fps)
    print(f"ByteTrack сохранен в {video_out_path} и {gif_out_path}")

def run_deepsort(video_path, model_path, output_dir):
    model = YOLO(model_path)
    tracker = DeepSort(max_age=15)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(3)), int(cap.get(4))

    filename = os.path.splitext(os.path.basename(video_path))[0]
    video_out_path = os.path.join(output_dir, f"{filename}_deepsort.mp4")
    gif_out_path = os.path.join(output_dir, f"{filename}.gif")
    os.makedirs(output_dir, exist_ok=True)
    out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    history = defaultdict(list)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        dets = []
        results = model(frame)[0]



        for box in results.boxes:
            conf = box.conf[0].cpu().item()
            if conf < 0.5:
                  continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item()
            cls = int(box.cls[0].cpu().item())
            dets.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        tracks = tracker.update_tracks(dets, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            tid = track.track_id
            color = get_color(tid)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            history[tid].append((cx, cy))
            for j in range(1, len(history[tid])):
                cv2.line(frame, history[tid][j - 1], history[tid][j], color, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)

    cap.release()
    out.release()

    VideoFileClip(video_out_path).write_gif(gif_out_path, fps=fps)
    print(f"DeepSORT сохранен в {video_out_path} и {gif_out_path}")
