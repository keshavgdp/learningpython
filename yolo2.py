import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("best (2).pt")

# Read input image paths
front_image_path = input("Enter the FRONT image path: ").strip()
back_image_path = input("Enter the BACK image path: ").strip()

# Step 1: Flip the front image horizontally (mirror flip)
front_image = cv2.imread(front_image_path)
flipped_front_image = cv2.flip(front_image, 1)  # 1 for horizontal flip

# Step 2: Run YOLO on both images (front and back)
front_results = model.predict(flipped_front_image, save=False)
back_results = model.predict(back_image_path, save=False)

# Function to extract fruit and pot coordinates (with bounding boxes)
def extract_fruit_and_pot_coordinates(results):
    objects = []  # List to store detected objects (fruits and pots)
    for result in results:
        boxes = result.boxes  # Get bounding box predictions
        for box in boxes:
            cls = int(box.cls[0])  # Get class index (fruit or pot)
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
            if cls in [1, 2]:  # Fruits (class 1 and 2)
                objects.append((cls, int(x1), int(y1), int(x2), int(y2)))  # Store relevant fruits
    return objects

# Extract coordinates of fruits in both images
front_fruits = extract_fruit_and_pot_coordinates(front_results)
back_fruits = extract_fruit_and_pot_coordinates(back_results)

# Function to crop the image to the detected fruits (only fruits)
def crop_to_fruits(image, fruits):
    # Find the combined bounding box for all detected fruits
    min_x = min(fruits, key=lambda fruit: fruit[1])[1]  # Minimum x-coordinate
    min_y = min(fruits, key=lambda fruit: fruit[2])[2]  # Minimum y-coordinate
    max_x = max(fruits, key=lambda fruit: fruit[3])[3]  # Maximum x-coordinate
    max_y = max(fruits, key=lambda fruit: fruit[4])[4]  # Maximum y-coordinate

    # Crop the image to include only the bounding box region containing the fruits
    cropped_image = image[min_y:max_y, min_x:max_x]
    return cropped_image, min_x, min_y

# Crop both front and back images to only include detected fruits
cropped_front_image, min_x_front, min_y_front = crop_to_fruits(flipped_front_image, front_fruits)
cropped_back_image, min_x_back, min_y_back = crop_to_fruits(cv2.imread(back_image_path), back_fruits)

# Function to draw bounding boxes around the detected fruits in the cropped images
def draw_bounding_boxes(image, fruits, min_x, min_y):
    for fruit in fruits:
        cls, x1, y1, x2, y2 = fruit
        # Adjust coordinates relative to the cropped image
        x1, y1, x2, y2 = x1 - min_x, y1 - min_y, x2 - min_x, y2 - min_y
        if cls in [1, 2]:  # Green box for fruits (class 1 and 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

# Draw bounding boxes on the cropped images (only fruits)
cropped_front_image_with_boxes = draw_bounding_boxes(cropped_front_image, front_fruits, min_x_front, min_y_front)
cropped_back_image_with_boxes = draw_bounding_boxes(cropped_back_image, back_fruits, min_x_back, min_y_back)

# Function to calculate the Intersection over Union (IoU) of two bounding boxes
def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate the intersection area
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate the area of both boxes
    area_box1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_box2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Calculate the union area
    union_area = area_box1 + area_box2 - inter_area

    # Calculate IoU
    iou = inter_area / union_area if union_area != 0 else 0
    return iou

# Function to compare fruits based on their IoU (with tolerance)
def compare_fruits(front_fruits, back_fruits, iou_threshold=0.5):
    matched_fruits = []  # To store fruits that are common in both images
    unique_fruits_front = []  # Fruits only in the front image
    unique_fruits_back = []  # Fruits only in the back image

    for front in front_fruits:
        front_cls, front_x1, front_y1, front_x2, front_y2 = front
        matched = False
        for back in back_fruits:
            back_cls, back_x1, back_y1, back_x2, back_y2 = back
            if front_cls == back_cls:  # Only compare fruits (class 1 or 2)
                iou = calculate_iou((front_x1, front_y1, front_x2, front_y2), (back_x1, back_y1, back_x2, back_y2))
                if iou >= iou_threshold:  # If IoU is above the threshold, consider them as matching
                    matched_fruits.append((front, back))  # Match found
                    matched = True
                    break
        
        if not matched:
            unique_fruits_front.append(front)  # Unique fruit in front image

    # Check for unique fruits in back image
    for back in back_fruits:
        back_cls, back_x1, back_y1, back_x2, back_y2 = back
        if not any(front_cls == back_cls and calculate_iou(
                (front_x1, front_y1, front_x2, front_y2), (back_x1, back_y1, back_x2, back_y2)) >= iou_threshold
                for front_x1, front_y1, front_x2, front_y2, front_cls in front_fruits):
            unique_fruits_back.append(back)  # Unique fruit in back image

    return matched_fruits, unique_fruits_front, unique_fruits_back

# Compare fruits in both images
matched_fruits, unique_fruits_front, unique_fruits_back = compare_fruits(front_fruits, back_fruits)

# Total fruit count = matched fruits + unique fruits in both images
total_fruit_count = len(matched_fruits) + len(unique_fruits_front) + len(unique_fruits_back)

# Output the total fruit count
print(f"Total Fruit Count: {total_fruit_count}")

# Display the cropped images with bounding boxes (only fruits detected)
cv2.imshow("Cropped Flipped Front Image with Fruits", cropped_front_image_with_boxes)
cv2.imshow("Cropped Back Image with Fruits", cropped_back_image_with_boxes)

cv2.waitKey(0)  # Wait for key press
cv2.destroyAllWindows()  # Close windows
