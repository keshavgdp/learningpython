from ultralytics import YOLO


model = YOLO("best (2).pt") 


front_image_path = input("Enter the FRONT image path: ").strip()
back_image_path = input("Enter the BACK image path: ").strip()


front_results = model.predict(front_image_path, save=False)
back_results = model.predict(back_image_path, save=False)


def extract_object_data(results):
    object_data = []  
    for result in results:  
        boxes = result.boxes 
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2 t
            width, height = x2 - x1, y2 - y1 
            cls = int(box.cls[0])  # Get class index 
            object_data.append((cls, center_x, center_y, width, height)) 
          ###Defining a Function to Extract Detected Objects###
    
    return object_data 


front_objects = extract_object_data(front_results)
back_objects = extract_object_data(back_results)


print("\n🔹 Detected Objects in Front Image:", front_objects)
print("\n🔹 Detected Objects in Back Image:", back_objects)
