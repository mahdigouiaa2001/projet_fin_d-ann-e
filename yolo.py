import os
import shutil
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import ultralytics

ultralytics.checks()

# Load our custom bottle detection deep learning model
model = YOLO("best1.pt")
img_path = "img_91.jpg"
results = model.predict(source=img_path, save=True)

# Extract detected bounding boxes
detected_boxes = []
for result in results:
    boxes = result.boxes.xyxy.tolist()
    detected_boxes.extend(boxes)

# Load the image
image = cv2.imread(img_path)
"""
# Create figure and axes
fig, ax = plt.subplots(1)

# Display the image
ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Iterate through the detected boxes
for box in detected_boxes:
    x_min, y_min, x_max, y_max = map(int, box)
    # Draw bounding box on the image
    rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

# Remove axis
ax.axis('off')"""

# Clear the "cropped_images" folder
save_dir = "cropped_images"
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir, exist_ok=True)

# Save cropped images and box coordinates to a text file
with open("box_coordinates.txt", "w") as f:
    for i, box in enumerate(detected_boxes):
        x_min, y_min, x_max, y_max = map(int, box)
        # Crop the region of interest from the image
        cropped_region = image[y_min:y_max, x_min:x_max]
        # Save the cropped region to the specified directory
        cv2.imwrite(os.path.join(save_dir, f"cropped_{i}.jpg"), cropped_region)
        # Write image name and box coordinates to the text file
        f.write(f"{save_dir}/cropped_{i}.jpg {x_min} {y_min} {x_max} {y_max}\n")

# Show plot
plt.show()

# Check the saved images in the directory
print("Cropped images saved in:", save_dir)
print("Box coordinates saved in: box_coordinates.txt")