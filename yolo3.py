import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("best (2).pt")

# Read input image paths
front_image_path = input("Enter the FRONT image path: ").strip()
back_image_path = input("Enter the BACK image path: ").strip()

# Step 1: Flip the front image horizontally (mirror flip)
front_image = cv2.imread(front_image_path)
back_image = cv2.imread(back_image_path)
flipped_front_image = cv2.flip(front_image, 1)

# Get image dimensions
front_height, front_width = flipped_front_image.shape[:2]
back_height, back_width = back_image.shape[:2]

# Step 2: Run YOLO on both images
front_results = model.predict(flipped_front_image, save=False)
back_results = model.predict(back_image, save=False)

def extract_fruit_coordinates(results, image_width, image_height):
    objects = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])  # Class ID
            conf = float(box.conf[0])  # Confidence score
            if conf < 0.5:  # Skip low-confidence detections
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
            # Normalize coordinates relative to image size
            x1 = x1 / image_width
            x2 = x2 / image_width
            y1 = y1 / image_height
            y2 = y2 / image_height
            
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            
            objects.append((cls, x1, y1, x2, y2, center_x, center_y, width, height, conf))
    return objects

# Extract normalized coordinates
front_fruits = extract_fruit_coordinates(front_results, front_width, front_height)
back_fruits = extract_fruit_coordinates(back_results, back_width, back_height)

def draw_boxes(image, fruits, image_width, image_height, title):
    for fruit in fruits:
        cls, x1, y1, x2, y2, _, _, _, _, conf = fruit
        # Convert normalized coordinates back to pixel values
        x1 = int(x1 * image_width)
        y1 = int(y1 * image_height)
        x2 = int(x2 * image_width)
        y2 = int(y2 * image_height)
        
        # Draw bounding box
        color = (0, 255, 0)  # Green color for bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Add class label and confidence score
        label = f"Class {cls} ({conf:.2f})"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Display the image
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Draw bounding boxes and labels on the front and back images
draw_boxes(flipped_front_image.copy(), front_fruits, front_width, front_height, "Front Image with Detections")
draw_boxes(back_image.copy(), back_fruits, back_width, back_height, "Back Image with Detections")

def match_fruits(front_fruits, back_fruits, distance_threshold=0.05, size_threshold=0.1):
    matched_fruits = []
    used_indices = set()
    
    # Match fruits based on their proximity in both x and y coordinates and size similarity
    for i, front_fruit in enumerate(front_fruits):
        front_cls, _, _, _, _, front_x, front_y, front_w, front_h, _ = front_fruit
        min_distance = float('inf')
        match_index = -1
        
        for j, back_fruit in enumerate(back_fruits):
            if j in used_indices:
                continue
            back_cls, _, _, _, _, back_x, back_y, back_w, back_h, _ = back_fruit
            distance = np.sqrt((front_x - back_x)**2 + (front_y - back_y)**2)
            size_diff = abs(front_w - back_w) + abs(front_h - back_h)
            
            # Match only if the class is the same, distance is small, and size is similar
            if front_cls == back_cls and distance < distance_threshold and size_diff < size_threshold:
                if distance < min_distance:
                    min_distance = distance
                    match_index = j
        
        if match_index != -1:
            matched_fruits.append(front_fruit)
            used_indices.add(match_index)
        else:
            # If no match is found, add the front fruit as a unique fruit
            matched_fruits.append(front_fruit)
    
    # Add unmatched back fruits
    for j, back_fruit in enumerate(back_fruits):
        if j not in used_indices:
            matched_fruits.append(back_fruit)
    
    return matched_fruits

# Match fruits between front and back images
matched_fruits = match_fruits(front_fruits, back_fruits)

# Count fruits by class
fruit_count = {}
for fruit in matched_fruits:
    cls = fruit[0]
    if cls in fruit_count:
        fruit_count[cls] += 1
    else:
        fruit_count[cls] = 1

# Print the total count of fruits by class
print("\nTotal Fruit Count by Class:")
for cls, count in fruit_count.items():
    print(f"Class {cls}: {count} fruits")
