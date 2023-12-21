import os
import numpy as np
import tensorflow as tf
from object_detection.utils import visualization_utils as visual_utils
from object_detection.utils import label_map_util
import matplotlib.pyplot as plt

model_path = 'trainedModels/trainedModel/saved_model'
label_map_path = 'dataset/label_map.pbtxt'
test_images_path = 'dataset/test'
output_dir = 'output'
input_image = 'plastic94.jpg'
multiple_detection = True
image_path = os.path.join(test_images_path, input_image)


def detect_objects(image_np, detection_model, multiple_detection):
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detection_model(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    if num_detections > 0:
        if multiple_detection:
            return detections
        else:
            max_score_idx = np.argmax(detections['detection_scores'])
            return {
                'detection_boxes': detections['detection_boxes'][max_score_idx],
                'detection_classes': detections['detection_classes'][max_score_idx],
                'detection_scores': detections['detection_scores'][max_score_idx],
                'num_detections': 1
            }
    else:
        return None


def save_detections(image_np, detections, category_index, image_file, multiple_detection):
    image_np_with_detections = image_np.copy()
    if multiple_detection:
        visual_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=len(detections['detection_boxes']),
            min_score_thresh=0.2,
            agnostic_mode=False
        )
    else:
        visual_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            np.expand_dims(detections['detection_boxes'], axis=0),
            np.expand_dims(detections['detection_classes'], axis=0),
            np.expand_dims(detections['detection_scores'], axis=0),
            category_index,
            use_normalized_coordinates=True,
            min_score_thresh=0.2,
            agnostic_mode=False
        )

    detection_scores = detections['detection_scores']
    detection_classes = np.squeeze(detections['detection_classes'])

    if np.isscalar(detection_classes):
        detection_classes = np.array([detection_classes])

    valid_indices = np.where(detection_scores > 0.2)[0]

    # Get the labels of the detected objects
    if multiple_detection:
        labels = [category_index[detection_classes]['name'] for detection_classes in
                  detections['detection_classes'][valid_indices]]
    else:
        labels = [category_index[detection_class]['name'] for detection_class in detection_classes[valid_indices]]

    # Counts the objects by class
    object_counts = {}
    for label in labels:
        object_counts[label] = object_counts.get(label, 0) + 1

    # Prints the counts for each class
    print(f"Objects found in {image_file}:")
    for label, count in object_counts.items():
        print(f"{label}: {count}")

    output_path = os.path.join(output_dir, f'annotated_{image_file}')
    plt.imsave(output_path, image_np_with_detections)


def main():
    detection_model = tf.saved_model.load(model_path)
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=5, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    image_np = plt.imread(image_path)

    detections = detect_objects(image_np, detection_model, multiple_detection)

    if detections is not None:
        save_detections(image_np, detections, category_index, input_image, multiple_detection)
    else:
        print(f"No objects detected in {input_image}.")

# def main():
#     detection_model = tf.saved_model.load(model_path)
#     label_map = label_map_util.load_labelmap(label_map_path)
#     categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=5, use_display_name=True)
#     category_index = label_map_util.create_category_index(categories)
#
#     test_image_files = [file for file in os.listdir(test_images_path) if file.lower().endswith('.jpg')]
#
#     for image_file in test_image_files:
#         image_path = os.path.join(test_images_path, image_file)
#         image_np = plt.imread(image_path)
#
#         detections = detect_objects(image_np, detection_model, multiple_detection)
#
#         if detections is not None:
#             save_detections(image_np, detections, category_index, image_file, multiple_detection)
#         else:
#             print(f"No objects detected in {image_file}.")


if __name__ == '__main__':
    main()
