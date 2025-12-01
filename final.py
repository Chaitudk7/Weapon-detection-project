import sys

try:
    import numpy as np
    import cv2  # OpenCV for video processing
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import io
    import gradio as gr
    from ultralytics import YOLO
    import winsound  # Windows beep sound (Windows only)
except Exception as e:
    print("Error importing required packages:", e)
    print("Please install dependencies with: pip install -r requirements.txt")
    sys.exit(1)

# Load your YOLO model
model = YOLO(r"C:\Users\HP\OneDrive\Desktop\FINAL\best.pt")

# Define function to process videos
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = "processed_video.mp4"
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height))

    weapon_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, save=False)
        result = results[0]

        for box in result.boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            label = f"{result.names[class_id]} {confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            weapon_detected = True

        out.write(frame)

    cap.release()
    out.release()

    # Play beep if a weapon is detected
    if weapon_detected:
        winsound.Beep(1000, 500)

    return output_video_path  # ✅ Return just the video path

# Function to process image uploads (optional placeholder if needed)
def predict_and_visualize(image_path):
    img = Image.open(image_path)
    frame = np.array(img)

    results = model.predict(source=frame, save=False)
    result = results[0]

    weapon_detected = False

    for box in result.boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
        confidence = box.conf[0]
        class_id = int(box.cls[0])
        label = f"{result.names[class_id]} {confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        weapon_detected = True

    if weapon_detected:
        winsound.Beep(1000, 500)

    return Image.fromarray(frame)  # Return image

# Function to handle both image and video uploads
def process_file(file_path):
    if file_path.lower().endswith((".jpg", ".jpeg", ".png")):
        return predict_and_visualize(file_path)  # Returns an image
    elif file_path.lower().endswith((".mp4", ".avi", ".mov")):
        return process_video(file_path)  # ✅ Return video path only
    else:
        return None  # Unsupported file format

# Function to process webcam frames
def process_webcam(frame):
    results = model.predict(source=frame, save=False)
    result = results[0]

    weapon_detected = False

    for box in result.boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
        confidence = box.conf[0]
        class_id = int(box.cls[0])
        label = f"{result.names[class_id]} {confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        weapon_detected = True

    if weapon_detected:
        winsound.Beep(1000, 500)

    return frame  # Return the processed frame

# Create the Gradio interface
file_upload_iface = gr.Interface(
    fn=process_file,
    inputs=gr.File(label="Video"),
    outputs=gr.Video(label="Processed Video"),
    title="Weapon Detection System",
    description="Upload a video to detect weapons. A beep sound will play if a weapon is detected.",
)

webcam_iface = gr.Interface(
    fn=process_webcam,
    inputs=gr.Image(label="Webcam Feed"),  # Webcam frame input
    outputs=gr.Image(label="Webcam Detection"),
    title="Weapon Detection via Webcam",
    description="Live weapon detection using your webcam.",
    live=True
)

# Combine both interfaces into a tabbed layout
gr.TabbedInterface([file_upload_iface, webcam_iface], ["File Upload", "Webcam"]).launch()
