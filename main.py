import argparse
import cv2
from utils.yolo_video import load_video, load_model, process_frame, save_log, count_id_switches

def calculate_direction_speed(log_data):
    speeds = {}
    for i in range(1, len(log_data)):
        curr = log_data[i]
        prev = log_data[i-1]
        if curr["track_id"] == prev["track_id"]:
            dx = curr["x1"] - prev["x1"]
            dy = curr["y1"] - prev["y1"]
            distance = ((dx ** 2 + dy ** 2) ** 0.5)
            speeds[curr["track_id"]] = distance if curr["track_id"] not in speeds else speeds[curr["track_id"]] + distance
    return speeds

def main(video_path, model_path, selected_classes=None, ground_truth_path=None):
    print(f"Processing video: {video_path}")
    print(f"Loading model: {model_path}")
    cap = load_video(video_path)
    print(f"Video opened: {cap.isOpened()}")
    model, class_names = load_model(model_path)
    log_data = []
    frame_count = 0
    objects_detected = False
    metrics_list = []
    total_fp = 0
    total_fn = 0
    total_gt = 0

    if selected_classes:
        selected_class_ids = [k for k, v in class_names.items() if v in selected_classes]
    else:
        selected_class_ids = None

    ground_truth = None
    if ground_truth_path:
        import json
        try:
            with open(ground_truth_path, 'r') as f:
                ground_truth = json.load(f)
        except Exception as e:
            print(f"Error loading ground truth: {e}")
            ground_truth = None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"End of video or error reading frame at frame {frame_count}")
            break
        frame_count += 1
        print(f"Processing frame {frame_count}")

        frame, frame_log, detected, metrics = process_frame(
            frame, model, class_names, selected_class_ids=selected_class_ids,
            ground_truth=ground_truth.get(str(frame_count), []) if ground_truth else [],
            frame_number=frame_count
        )
        log_data.extend(frame_log)
        if detected:
            objects_detected = True
        if metrics:
            metrics_list.append(metrics)
            # Accumulate FP, FN, and GT for MOTA
            if ground_truth and str(frame_count) in ground_truth:
                frame_gt = ground_truth[str(frame_count)]
                total_gt += len(frame_gt)
                frame_fp = len(frame_log) - metrics["precision"] * len(frame_log)  # Approximate FP
                frame_fn = len(frame_gt) - metrics["recall"] * len(frame_gt)  # Approximate FN
                total_fp += frame_fp
                total_fn += frame_fn

        cv2.imshow("Object Detection and Tracking", frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User interrupted with 'q'")
            break

    if metrics_list:
        avg_precision = sum(m["precision"] for m in metrics_list) / len(metrics_list)
        avg_recall = sum(m["recall"] for m in metrics_list) / len(metrics_list)
        avg_mAP = sum(m["mAP"] for m in metrics_list) / len(metrics_list)
        print(f"Global Metrics - Precision: {avg_precision:.2f}, Recall: {avg_recall:.2f}, mAP: {avg_mAP:.2f}")

    # Calculate tracking performance
    id_switches = count_id_switches(log_data)
    print(f"Number of ID switches: {id_switches}")

    # Calculate MOTA
    if total_gt > 0:
        mota = 1 - (total_fp + total_fn + id_switches) / total_gt
        print(f"MOTA (Tracking Accuracy): {mota:.2f}")
    else:
        print("MOTA cannot be calculated: No ground truth provided or no objects detected.")

    speeds = calculate_direction_speed(log_data)
    if speeds:
        print("Approximate speeds (pixels per frame):", {k: v/frame_count for k, v in speeds.items()})

    if not objects_detected:
        print("No objects detected in the video.")
    else:
        print(f"Objects detected: {len(log_data)} detections")

    save_log(log_data)
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="data/voiture.mp4", help="Path to the video")
    parser.add_argument("--model", type=str, default="model/yolov8n.pt", help="Path to the model")
    parser.add_argument("--classes", type=str, nargs='+', help="Classes to detect (e.g., person car traffic_light)")
    parser.add_argument("--ground-truth", type=str, help="Path to ground truth annotations (optional)")
    args = parser.parse_args()
    main(args.video, args.model, args.classes, args.ground_truth)