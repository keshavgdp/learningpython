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

print(f"Front image dimensions: {front_width}x{front_height}")
print(f"Back image dimensions: {back_width}x{back_height}")

# Step 2: Run YOLO on both images
front_results = model.predict(flipped_front_image, save=False)
back_results = model.predict(back_image, save=False)

def extract_fruit_coordinates(results, image_width, image_height):
    objects = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < 0.5:
                print(f"Skipping low-confidence detection: Class {cls}, Confidence {conf:.2f}")
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # Normalize coordinates relative to image size
            x1 = x1 / image_width
            x2 = x2 / image_width
            y1 = y1 / image_height
            y2 = y2 / image_height
            
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            
            objects.append((cls, center_x, center_y, width, height))
            print(f"Extracted fruit: Class {cls}, Center ({center_x:.3f}, {center_y:.3f}), Size: {width:.3f}x{height:.3f}")
    return objects

# Extract normalized coordinates
front_fruits = extract_fruit_coordinates(front_results, front_width, front_height)
back_fruits = extract_fruit_coordinates(back_results, back_width, back_height)

print("\nFront fruits:")
for i, fruit in enumerate(front_fruits):
    cls, x, y, w, h = fruit
    print(f"Fruit {i}: Class {cls}, Center ({x:.3f}, {y:.3f}), Size: {w:.3f}x{h:.3f}")

print("\nBack fruits:")
for i, fruit in enumerate(back_fruits):
    cls, x, y, w, h = fruit
    print(f"Fruit {i}: Class {cls}, Center ({x:.3f}, {y:.3f}), Size: {w:.3f}x{h:.3f}")

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def match_fruits(front_fruits, back_fruits, iou_threshold=0.1, distance_threshold=0.2):
    """
    Match fruits between front and back images using IoU and distance.
    """
    matched_fruits = []
    used_indices = set()
    
    print("\nMatching fruits:")
    for i, front_fruit in enumerate(front_fruits):
        front_cls, front_x, front_y, front_w, front_h = front_fruit
        min_distance = float('inf')
        match_index = -1
        
        print(f"\nMatching front fruit {i}: Class {front_cls}, Center ({front_x:.3f}, {front_y:.3f})")
        for j, back_fruit in enumerate(back_fruits):
            if j in used_indices:
                print(f"Skipping back fruit {j} (already matched)")
                continue
            back_cls, back_x, back_y, back_w, back_h = back_fruit
            if front_cls != back_cls:
                print(f"Skipping back fruit {j} (different class)")
                continue
            
            # Calculate IoU
            iou = calculate_iou(
                (front_x - front_w / 2, front_y - front_h / 2, front_x + front_w / 2, front_y + front_h / 2),
                (back_x - back_w / 2, back_y - back_h / 2, back_x + back_w / 2, back_y + back_h / 2)
            )
            print(f"Comparing with back fruit {j}: Class {back_cls}, Center ({back_x:.3f}, {back_y:.3f}), IoU: {iou:.3f}")
            
            if iou >= iou_threshold:
                print(f"Match found: IoU {iou:.3f} >= threshold {iou_threshold}")
                match_index = j
                break
            else:
                # If IoU is low, check distance
                distance = np.sqrt((front_x - back_x)**2 + (front_y - back_y)**2)
                print(f"Distance: {distance:.3f}")
                if distance < min_distance and distance < distance_threshold:
                    min_distance = distance
                    match_index = j
        
        if match_index != -1:
            matched_fruits.append((front_cls, front_x, front_y, front_w, front_h))
            used_indices.add(match_index)
            print(f"Matched front fruit {i} with back fruit {match_index}")
        else:
            print(f"No match found for front fruit {i}")
            matched_fruits.append(front_fruit)
    
    # Add unmatched back fruits
    for j, back_fruit in enumerate(back_fruits):
        if j not in used_indices:
            print(f"Added unmatched back fruit {j}")
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
