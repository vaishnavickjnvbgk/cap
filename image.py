import streamlit as st
import os
from PIL import Image
import subprocess
import shutil
import cv2
import torch
import numpy as np
import subprocess
import streamlit as st
from PIL import Image as PILImage
from io import BytesIO
import os


import os
import shutil
import subprocess
import streamlit as st
from PIL import Image

# Folder paths
upload_folder = "uploads"
result_folder = "uploads/results"
labels_folder = "uploads/results/labels"
image_name = "image.jpg"
result_image_name = "result_image.jpg"
label_name = "image.txt"

# Create the folders if they don't exist
os.makedirs(upload_folder, exist_ok=True)
os.makedirs(result_folder, exist_ok=True)
os.makedirs(labels_folder, exist_ok=True)

# Function to empty a folder
def empty_folder(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            st.error(f"Error deleting file {file_name}: {str(e)}")

# Empty the upload and result folders before starting
empty_folder(upload_folder)
empty_folder(result_folder)
empty_folder(labels_folder)

# Title of the Streamlit app
st.title("Image Upload and YOLOv5 Detection")

# Upload image or take a photo
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
camera_image = st.camera_input("Take a photo")

if uploaded_file is not None and camera_image is not None:
    st.warning("Please upload an image or take a photo, not both.")
else:
    image = None
    # Determine image source
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    elif camera_image is not None:
        image = Image.open(camera_image)

    if image:
        # Save the image with a fixed name in the 'uploads' folder
        image_path = os.path.join(upload_folder, image_name)

        # Save the new uploaded or captured image
        try:
            image.save(image_path)
            st.image(image, caption="Uploaded or Captured Image.", use_column_width=True)
            st.write(f"New image saved at: {image_path}")
        except Exception as e:
            st.error(f"Error saving image: {str(e)}")
        if not os.path.exists("yolov5"):
            st.write("Cloning YOLOv5 repository...")
            subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
        
        if os.path.exists(image_path):
            with st.spinner("Running YOLOv5 detection..."):
                # YOLOv5 detection command
                command = [
                    "python", "yolov5/detect.py", "--weights", "best.pt", "--img", "640", "--conf", "0.10",
                    "--source", image_path, "--save-txt", "--save-conf", "--project", upload_folder,
                    "--name", "results", "--exist-ok"
                ]

                try:
                    # Run YOLOv5 detection
                    subprocess.run(command, check=True, capture_output=True, text=True)
                    st.success("YOLOv5 detection completed successfully.")
                except subprocess.CalledProcessError as e:
                    st.error(f"Error running YOLOv5 detection: {e.stderr}")

                # Paths to the result image and label file
                detected_image_path = os.path.join(result_folder, "image0.jpg")
                result_image_path = os.path.join(result_folder, result_image_name)
                result_label_path = os.path.join(upload_folder, "results", "labels", "image0.txt")  # Path where YOLOv5 outputs label file
                updated_label_path = os.path.join(result_folder, label_name)  # Our consistent label file in results folder

                if os.path.exists(detected_image_path):
                    try:
                        # Move the result image to the result folder
                        os.rename(detected_image_path, result_image_path)
                        st.image(result_image_path, caption="Detection Result", use_column_width=True)
                        st.write(f"Detection image saved at: {result_image_path}")
                    except Exception as e:
                        st.error(f"Error handling result image: {str(e)}")

                    # Ensure the label file exists in the correct path
                    if os.path.exists(result_label_path):
                        try:
                            # Print a message indicating that the label file is being moved
                            st.write(f"Moving label file from {result_label_path} to {updated_label_path}")

                            # Move the label file to the result folder
                            shutil.move(result_label_path, updated_label_path)
                            st.write(f"Label file moved to: {updated_label_path}")
                        except Exception as e:
                            st.error(f"Error handling label file: {str(e)}")
                    else:
                        st.error("Label file not found in the expected location.")
                else:
                    st.error("Detected image not found.")
        else:
            st.error(f"Image not found at: {image_path}")


def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))


def classify_capsicum(image, bounding_box):
    x1, y1, x2, y2 = bounding_box
    roi = image[y1:y2, x1:x2]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    ripe_lower = np.array([20, 40, 40])
    ripe_upper = np.array([85, 255, 255])

    unripe_lower = np.array([35, 40, 40])
    unripe_upper = np.array([85, 255, 255])

    ripe_mask = cv2.inRange(hsv_roi, ripe_lower, ripe_upper)
    unripe_mask = cv2.inRange(hsv_roi, unripe_lower, unripe_upper)

    total_pixels = roi.shape[0] * roi.shape[1]
    ripe_percentage = (cv2.countNonZero(ripe_mask) / total_pixels) * 100
    unripe_percentage = (cv2.countNonZero(unripe_mask) / total_pixels) * 100

    if ripe_percentage > 20:
        return "ripe"
    elif unripe_percentage > 50:
        return "unripe"
    else:
        return "unknown"


def grade_capsicum(capsicum_height, capsicum_width, health_status):
    if health_status == "healthy":
        if capsicum_height > 200 and capsicum_width > 220:
            
            return "Grade A"
        else:
            return "Grade B"
    else:
        return "Grade C"






def make_background_white(image_path, detections_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    try:
        with open(detections_path, 'r') as f:
            detections = f.readlines()
    except FileNotFoundError:
        print(f"Error: Detection file not found at {detections_path}")
        return
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    bounding_boxes = []
   

    for detection in detections:
        detection = detection.strip().split()
        x_center, y_center, width, height = map(float, detection[1:5])
        x_center *= image.shape[1]
        y_center *= image.shape[0]
        width *= image.shape[1]
        height *= image.shape[0]
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        bounding_boxes.append((x1, y1, x2, y2))
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

    mask_inv = cv2.bitwise_not(mask)

    white_background = np.ones_like(image) * 255
    image_with_white_bg = cv2.bitwise_and(image, image, mask=mask)
    white_background = cv2.bitwise_and(white_background, white_background, mask=mask_inv)
    final_image = cv2.add(image_with_white_bg, white_background)


    cv2.imwrite(output_path, final_image)
    print(f"Saved image with white background to {output_path}")

    grade_counts = {"Grade A": 0, "Grade B": 0, "Grade C": 0}

    

    gray_image_path = output_path.replace('.jpg', '_gray.jpg')
    gray_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(gray_image_path, gray_image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray_image = clahe.apply(gray_image)

    blurred_images = [cv2.GaussianBlur(enhanced_gray_image, (k, k), 0) for k in [3, 5, 7]]
    edges_list = [cv2.Canny(blurred, 50, 150) for blurred in blurred_images]
    combined_edges = np.bitwise_or.reduce(edges_list)

    kernel = np.ones((3, 3), np.uint8)
    enhanced_edges = cv2.dilate(combined_edges, kernel, iterations=1)
    enhanced_edges = cv2.erode(enhanced_edges, kernel, iterations=1)

    edges_image_path = gray_image_path.replace('_gray.jpg', '_edges.jpg')
    cv2.imwrite(edges_image_path, enhanced_edges)

    color_palette_hex = ["d99a50", "e5b47c", "a35515", "ba6e18", "be7128", "cd8229", "c2823c", "b26c2b", "e5b468", "a3672d"]
    color_palette_bgr = [hex_to_bgr(color) for color in color_palette_hex]

    palette_mask = np.zeros_like(mask)
    for color in color_palette_bgr:
        color_mask = cv2.inRange(image, np.array(color) - 10, np.array(color) + 10)
        palette_mask = cv2.bitwise_or(palette_mask, color_mask)

    final_edges_palette = cv2.bitwise_and(enhanced_edges, enhanced_edges, mask=palette_mask)
    final_edges_palette_path = edges_image_path.replace('_edges.jpg', '_edges_palette.jpg')
    cv2.imwrite(final_edges_palette_path, final_edges_palette)
    
    for (x1, y1, x2, y2) in bounding_boxes:
        region_edges = final_edges_palette[y1:y2, x1:x2]
        total_region_pixels = region_edges.shape[0] * region_edges.shape[1]
        non_zero_edges = cv2.countNonZero(region_edges)
        edge_percentage = (non_zero_edges / total_region_pixels) * 100

        spoilt_label = "healthy" if edge_percentage < 0.01 else "spoilt"
        color = (0, 0, 255) if spoilt_label == "spoilt" else (0, 255, 0)

        capsicum_height = y2 - y1
        capsicum_width = x2 - x1

        if spoilt_label == "healthy":
            ripeness_label = classify_capsicum(image, (x1, y1, x2, y2))
            label = f"{spoilt_label}, {ripeness_label}"
            grade = grade_capsicum(capsicum_height, capsicum_width, spoilt_label)
            grade_counts[grade] += 1
        else:
            label = spoilt_label
            grade = grade_capsicum(capsicum_height, capsicum_width, spoilt_label)
            grade_counts[grade] += 1

        grade = grade_capsicum(capsicum_height, capsicum_width, spoilt_label)
        label += f", {grade}"

        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    labeled_image_path = output_path.replace('.jpg', '_labeled.jpg')
    cv2.imwrite(labeled_image_path, image)
    print(f"Saved labeled image to {labeled_image_path}")

    

    st.write("Capsicum Grade Counts:")
    for grade, count in grade_counts.items():
       st.write(f"{grade}: {count}")

# Get paths to image and label file from the results folder
image_path = "uploads/image.jpg"
detections_path = "uploads/results/labels/image.txt"
output_path = "uploads/results/image_with_white_bg.jpg"

# Call the function to process the image
make_background_white(image_path, detections_path, output_path)

# Define the path for the processed image
labeled_image_path = "uploads/results/image_with_white_bg_labeled.jpg"

# Check if the labeled image exists before displaying
if os.path.exists(labeled_image_path):
    # Display the image
    st.image(labeled_image_path, caption="Image with Labeled Detection", use_column_width=True)
    st.write(f"Image with labels displayed from: {labeled_image_path}")
else:
    st.error(f"Labeled image not found at {labeled_image_path}")