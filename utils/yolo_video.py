import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import pandas as pd
from sklearn.metrics import precision_score, recall_score, average_precision_score

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    intersection = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def load_video(video_path):
    print(f"Attempting to load video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"OpenCV failed to open video: {video_path}")
        raise ValueError(f"Error: Unable to load video {video_path}.")
    return cap

def load_model(model_path):
    model = YOLO(model_path)
    class_names = model.names
    return model, class_names

def process_frame(frame, model, class_names, selected_class_ids=None, ground_truth=None, frame_number=None):
    if selected_class_ids is None:
        selected_class_ids = list(class_names.keys())

    results = model.track(frame, classes=selected_class_ids, persist=True, conf=0.5)
    detections = []
    objects_detected = False
    
    detected_boxes = []
    detected_classes = []
    detected_confidences = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            if box.id is not None:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())  # Convert to scalar float
                track_id = int(box.id[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                object_class = class_names.get(class_id, "unknown")
                
                objects_detected = True
                print(f"Alert: {object_class} detected at {datetime.now()}")
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, f"{object_class} ID: {track_id}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                detections.append({
                    "frame_number": frame_number,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "track_id": track_id,
                    "class": object_class,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": conf  # Now a float
                })
                detected_boxes.append([x1, y1, x2, y2])
                detected_classes.append(class_id)
                detected_confidences.append(conf)

    if not objects_detected:
        cv2.putText(frame, "No objects detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    metrics = {}
    if ground_truth:
        gt_boxes = [[gt["x1"], gt["y1"], gt["x2"], gt["y2"]] for gt in ground_truth]
        gt_classes = [gt["class_id"] for gt in ground_truth]
        
        iou_threshold = 0.5
        tp, fp, fn = 0, 0, len(gt_boxes)
        matched_gt = set()

        for det_idx, (det_box, det_class, det_conf) in enumerate(zip(detected_boxes, detected_classes, detected_confidences)):
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, (gt_box, gt_class) in enumerate(zip(gt_boxes, gt_classes)):
                if gt_idx in matched_gt:
                    continue
                if det_class != gt_class:
                    continue
                iou = calculate_iou(det_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                tp += 1
                fn -= 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if detected_confidences:
            sorted_indices = np.argsort(detected_confidences)[::-1]
            precisions = []
            recalls = []
            tp_cumsum = 0
            fp_cumsum = 0
            for idx in sorted_indices:
                det_box = detected_boxes[idx]
                det_class = detected_classes[idx]
                best_iou = 0
                best_gt_idx = -1
                for gt_idx, (gt_box, gt_class) in enumerate(zip(gt_boxes, gt_classes)):
                    if gt_idx in matched_gt:
                        continue
                    if det_class != gt_class:
                        continue
                    iou = calculate_iou(det_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                if best_iou >= iou_threshold:
                    tp_cumsum += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp_cumsum += 1
                prec = tp_cumsum / (tp_cumsum + fp_cumsum) if (tp_cumsum + fp_cumsum) > 0 else 0
                rec = tp_cumsum / len(gt_boxes) if len(gt_boxes) > 0 else 0
                precisions.append(prec)
                recalls.append(rec)
            mAP = np.mean(precisions) if precisions else 0
        else:
            mAP = 0

        metrics = {"precision": precision, "recall": recall, "mAP": mAP}

    return frame, detections, objects_detected, metrics

def save_log(log_data, output_path="tracking_log.csv"):
    if log_data:
        df = pd.DataFrame(log_data)
        df.to_csv(output_path, index=False)
        print(f"Log file generated: {output_path}")
    else:
        print("No objects detected, no log file generated.")

def count_id_switches(log_data):
    switches = 0
    prev_ids = set()
    for i in range(1, len(log_data)):
        curr_id = log_data[i]["track_id"]
        if curr_id not in prev_ids and prev_ids:
            switches += 1
        prev_ids = {curr_id}
    return switches