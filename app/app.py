import streamlit as st
import cv2
import tempfile
import os
import sys
import pandas as pd
import time

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f4f8;
        padding: 20px;
        border-radius: 10px;
    }
    .stApp {
        background-color: #e6eef6;
    }
    .stHeader {
        color: #1e3a8a;
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        padding: 10px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stSubheader {
        color: #2b6cb0;
        font-size: 1.5em;
        margin-top: 15px;
    }
    .stWarning {
        background-color: #fef2f2;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #dc2626;
    }
    .stButton>button {
        background-color: #2b6cb0;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #1e40af;
    }
    .stSelectbox div {
        background-color: #ffffff;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom function to display a progress bar
def show_progress(status_text):
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)  # Simulate processing
        progress_bar.progress(i + 1)
    st.success(status_text)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.yolo_video import load_video, load_model, process_frame, save_log, count_id_switches

def main():
    # Decorative header
    st.markdown('<div class="stHeader">Object Detection and Tracking</div>', unsafe_allow_html=True)
    
    # User interface
    st.markdown('<div class="stSubheader">Select Objects to Detect</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        available_classes = list(load_model("model/yolov8n.pt")[1].values())
        selected_classes = st.multiselect(
            "Choose the classes to detect",
            options=available_classes,
            default=["person", "car", "traffic light"],
            help="Select the objects you want to detect in the video."
        )
    with col2:
        st.image("https://via.placeholder.com/100x100.png?text=Logo", use_container_width=True)  # Placeholder for logo
    
    # Check if selected classes are valid
    invalid_classes = [cls for cls in selected_classes if cls not in available_classes]
    if invalid_classes:
        st.error(f"The following objects are not supported: {', '.join(invalid_classes)}. Choose from the available options.")
        return
    
    selected_class_ids = [k for k, v in load_model("model/yolov8n.pt")[1].items() if v in selected_classes]
    
    if not selected_classes:
        st.warning("Please select at least one class for detection.")
        return
    
    ground_truth_file = st.file_uploader("Upload ground truth annotations (JSON)", type=["json"])
    ground_truth = None
    if ground_truth_file:
        import json
        ground_truth = json.load(ground_truth_file)
    
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
    
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        
        cap = load_video(tfile.name)
        stframe = st.empty()
        status = st.empty()
        status.write("Processing video in progress...")
        show_progress("Video processed successfully!")
        
        st.markdown('<div class="stSubheader">Real-Time Detections</div>', unsafe_allow_html=True)
        table_placeholder = st.empty()
        log_data = []
        table_data = []
        objects_detected = False
        metrics_list = []
        total_fp = 0
        total_fn = 0
        total_gt = 0
        frame_count = 0
        detected_classes = set()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            frame, frame_log, detected, metrics = process_frame(
                frame, load_model("model/yolov8n.pt")[0], load_model("model/yolov8n.pt")[1], selected_class_ids=selected_class_ids,
                ground_truth=ground_truth.get(str(frame_count), []) if ground_truth else [],
                frame_number=frame_count
            )
            log_data.extend(frame_log)
            if detected:
                objects_detected = True
                for log in frame_log:
                    detected_classes.add(log["class"])
                    table_data.append({
                        "Alert": f"Alert: {log['class']} detected",
                        "Object ID": log["track_id"],
                        "Class": log["class"],
                        "X1": round(log["x1"], 2),
                        "Y1": round(log["y1"], 2),
                        "X2": round(log["x2"], 2),
                        "Y2": round(log["y2"], 2),
                        "Time": log["timestamp"],
                        "Confidence": round(log["confidence"], 2)
                    })
                table_df = pd.DataFrame(table_data)
                table_placeholder.dataframe(table_df, use_container_width=True)
            
            if metrics:
                metrics_list.append(metrics)
                if ground_truth and str(frame_count) in ground_truth:
                    frame_gt = ground_truth[str(frame_count)]
                    total_gt += len(frame_gt)
                    frame_fp = len(frame_log) - metrics["precision"] * len(frame_log)
                    frame_fn = len(frame_gt) - metrics["recall"] * len(frame_gt)
                    total_fp += frame_fp
                    total_fn += frame_fn
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, use_container_width=True)
        
        # Display detection table with message for non-detected classes
        table_df = pd.DataFrame(table_data)
        table_placeholder.dataframe(table_df, use_container_width=True)
        non_detected_classes = [cls for cls in selected_classes if cls not in detected_classes]
        if non_detected_classes:
            st.warning(f"The following objects were not detected: {', '.join(non_detected_classes)}.")
        if not objects_detected:
            st.warning("No objects detected in the video.")
        
        st.markdown('<div class="stSubheader">Filter Detections</div>', unsafe_allow_html=True)
        filter_class = st.selectbox("Filter by class", options=["All"] + selected_classes, index=0)
        if filter_class != "All":
            filtered_data = [d for d in table_data if d["Class"] == filter_class]
        else:
            filtered_data = table_data
        filtered_df = pd.DataFrame(filtered_data)
        st.dataframe(filtered_df, use_container_width=True)
        
        if filtered_data:
            csv = filtered_df.to_csv(index=False)
            st.download_button("Download table", data=csv, file_name="detection_table.csv")
        
        if metrics_list:
            avg_precision = sum(m["precision"] for m in metrics_list) / len(metrics_list)
            avg_recall = sum(m["recall"] for m in metrics_list) / len(metrics_list)
            avg_mAP = sum(m["mAP"] for m in metrics_list) / len(metrics_list)
            st.write(f"Metrics - Precision: {avg_precision:.2f}, Recall: {avg_recall:.2f}, mAP: {avg_mAP:.2f}")
        
        id_switches = count_id_switches(log_data)
        st.write(f"ID Switches: {id_switches}")
        
        if total_gt > 0:
            mota = 1 - (total_fp + total_fn + id_switches) / total_gt
            st.write(f"MOTA: {mota:.2f}")
        else:
            st.write("MOTA cannot be calculated: no ground truth provided.")
        
        if log_data:
            save_log(log_data, "tracking_log_web.csv")
            st.download_button("Download log", data=open("tracking_log_web.csv", "rb"), file_name="tracking_log_web.csv")
        
        cap.release()
        os.remove(tfile.name)

if __name__ == "__main__":
    main()