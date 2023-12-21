import os
import xml.etree.ElementTree as ET
import csv

# Path to the directory containing the XML annotations
annotations_dir = "../dataset/annotations"

# Output CSV file path
output_csv_file = "../dataset/annotations/annotations.csv"

# List to store annotation data
annotations_data = []

# Loop through each XML file in the annotations directory
for xml_file in os.listdir(annotations_dir):
    if xml_file.endswith(".xml"):
        xml_path = os.path.join(annotations_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find("filename").text
        for obj in root.iter("object"):
            class_name = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            annotations_data.append([filename, class_name, xmin, ymin, xmax, ymax])

# Write the annotation data to a CSV file
with open(output_csv_file, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["filename", "class", "xmin", "ymin", "xmax", "ymax"])
    writer.writerows(annotations_data)
