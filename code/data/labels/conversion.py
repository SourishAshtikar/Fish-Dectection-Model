import csv
import os
from collections import defaultdict

# Read CSV
with open('detections.csv', 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)

# Group by image id
image_data = defaultdict(list)
for row in data:
    image_data[row['ImageID']].append(row)

# Create output folder
os.makedirs('labels', exist_ok=True)

for image_id, objects in image_data.items():
    txt_filename = image_id + '.txt'
    txt_path = os.path.join('labels', txt_filename)
    with open(txt_path, 'w') as f:
        for obj in objects:
            # Assign all to class 0
            class_id = 0

            xmin = float(obj['XMin'])
            xmax = float(obj['XMax'])
            ymin = float(obj['YMin'])
            ymax = float(obj['YMax'])

            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            box_width = xmax - xmin
            box_height = ymax - ymin

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

print("✅ Done. Check 'labels/' folder for YOLO files.")
