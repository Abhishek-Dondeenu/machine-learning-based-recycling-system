import os
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
from object_detection.utils import visualization_utils as visual_utils
from object_detection.utils import label_map_util
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix as cm

model_path = 'trainedModels/trainedModel/saved_model'
label_map_path = 'dataset/label_map.pbtxt'
test_images_path = 'dataset/test'
output_dir = 'output'
multiple_detection = True


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

def calculate_iou(detection_boxes, true_labels, iou_thresholds):
    iou_list = []
    for iou_threshold in iou_thresholds:
        true_positive = 0
        false_positive = 0
        false_negative = 0

        for i, box in enumerate(detection_boxes):
            iou = calculate_single_iou(box, true_labels[i])
            if iou >= iou_threshold:
                true_positive += 1
            else:
                false_positive += 1

        false_negative = true_labels.sum() - true_positive

        iou_value = true_positive / (true_positive + false_positive + false_negative)
        iou_list.append(iou_value)

    return iou_list


def calculate_single_iou(box, true_label):
    intersection = np.sum(np.logical_and(box, true_label))
    union = np.sum(np.logical_or(box, true_label))

    if union == 0:
        return 0

    return intersection / union


def calculate_classification_accuracy(true_labels, predicted_labels, num_classes):
    correct_predictions = (true_labels == predicted_labels)
    class_accuracy = {}
    overall_accuracy = sum(correct_predictions) / len(true_labels)

    for class_id in range(1, num_classes + 1):
        class_mask = (true_labels == class_id)
        class_correct_predictions = correct_predictions[class_mask]
        class_accuracy[class_id] = sum(class_correct_predictions) / class_mask.sum()

    return overall_accuracy, class_accuracy


def plot_confusion_matrix(confusion_matrix, class_names, output_file):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(output_file)
    plt.close()


def calculate_metrics(true_labels, predicted_labels, num_classes, detections, iou_thresholds=None):
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=5, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    class_labels = {class_id: category_index[class_id]['name'] for class_id in range(1, num_classes + 1)}

    if iou_thresholds is None:
        iou_thresholds = [0.5]

    if len(true_labels) != len(detections):
        raise ValueError("Found input variables with inconsistent numbers of samples.")

    confusion_matrix_data = cm(true_labels, predicted_labels)

    precision = {}
    recall = {}
    average_precision = {}
    f1_score = {}
    iou = {}

    for class_id in range(1, num_classes + 1):
        binary_true_labels = (true_labels == class_id).astype(int)

        if binary_true_labels.sum() > 0:
            if detections is None:
                detection_scores = []
            else:
                detection_scores = [
                    detection['detection_scores'][class_id - 1] if (
                        detection is not None and 'detection_scores' in detection and class_id - 1 < len(
                            detection['detection_scores'])) else 0 for detection in detections
                ]

            precision[class_id], recall[class_id], _ = precision_recall_curve(binary_true_labels,
                                                                              detection_scores,
                                                                              pos_label=1)
            average_precision[class_id] = average_precision_score(binary_true_labels,
                                                                  detection_scores)

            # Calculate F1-score
            denominator = precision[class_id] + recall[class_id]
            precision_class = np.nan_to_num(precision[class_id], nan=0)
            recall_class = np.nan_to_num(recall[class_id], nan=0)
            is_invalid = np.logical_or(np.isnan(denominator), denominator == 0)
            f1_score[class_id] = np.where(is_invalid, 0, 2 * (precision_class * recall_class) / (denominator + 1e-9))

            # Calculate IoU
            iou[class_id] = calculate_iou([detection['detection_boxes'][class_id - 1] if (
                detection is not None and 'detection_boxes' in detection and class_id - 1 < len(
                    detection['detection_boxes'])) else np.zeros((4,)) for detection in detections],
                binary_true_labels,
                iou_thresholds)

        else:
            precision[class_id] = 0
            recall[class_id] = 0
            average_precision[class_id] = 0
            f1_score[class_id] = 0
            iou[class_id] = [0] * len(iou_thresholds)

    mean_average_precision_labels = {class_labels[class_id]: ap for class_id, ap in average_precision.items()}

    return confusion_matrix_data, precision, recall, mean_average_precision_labels, f1_score, iou


def evaluate_model(detection_model, test_image_files, category_index, num_classes, iou_thresholds=None):
    if iou_thresholds is None:
        iou_thresholds = [0.5]

    true_labels = []
    predicted_labels = []
    detections_list = []

    for image_file in test_image_files:

        image_path = os.path.join(test_images_path, image_file)
        image_np = plt.imread(image_path)

        detections = detect_objects(image_np, detection_model, multiple_detection)

        detections_list.append(detections)

        if detections is not None:
            predicted_label = detections['detection_classes'][0] if detections['num_detections'] > 0 else -1
        else:
            predicted_label = -1

        base_name = os.path.splitext(image_file)[0]

        class_label = ''.join(filter(str.isalpha, base_name)).lower()
        class_label_id = 0

        if class_label.lower() == 'cardboard':
            class_label_id = 1
        elif class_label.lower() == 'glass':
            class_label_id = 2
        elif class_label.lower() == 'metal':
            class_label_id = 3
        elif class_label.lower() == 'paper':
            class_label_id = 4
        elif class_label.lower() == 'plastic':
            class_label_id = 5

        if class_label_id in category_index:
            true_label = category_index[class_label_id]['id']
        else:
            print(f"Warning: Class label '{class_label}' not found in category_index dictionary.")
            continue

        predicted_labels.append(predicted_label)
        true_labels.append(true_label)

        if detections is not None:
            save_detections(image_np, detections, category_index, image_file, multiple_detection)

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    overall_accuracy, class_accuracy = calculate_classification_accuracy(true_labels, predicted_labels, num_classes)

    confusion_matrix, precision, recall, mean_average_precision, f1_score, iou = calculate_metrics(true_labels,
                                                                                                   predicted_labels,
                                                                                                   num_classes,
                                                                                                   detections_list,
                                                                                                   iou_thresholds)

    print("Mean Average Precision (mAP):", mean_average_precision)
    for class_id, ap in precision.items():
        class_name = category_index[class_id]['name']
        print(f"Class - {class_name} - Average Precision (AP):", ap)
        print(f"Class - {class_name} - F1-score:", f1_score[class_id])
        for i, iou_threshold in enumerate(iou_thresholds):
            print(f"Class - {class_name} - IoU @ {iou_threshold}: {iou[class_id][i]}")

    print("Overall Accuracy:", overall_accuracy)
    for class_id, acc in class_accuracy.items():
        class_name = category_index[class_id]['name']
        print(f"Class - {class_name} - Accuracy: {acc}")

    return confusion_matrix, precision, recall, mean_average_precision, f1_score, iou, overall_accuracy, class_accuracy

def main(multiple_detection):

    detection_model = tf.saved_model.load(model_path)
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=5, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    test_image_files = [file for file in os.listdir(test_images_path) if file.lower().endswith('.jpg')]

    num_classes = 5

    confusion_matrix, precision, recall, mean_average_precision, f1_score, iou, overall_accuracy, class_accuracy = evaluate_model(
        detection_model,
        test_image_files,
        category_index,
        num_classes)

    for image_file in test_image_files:
        image_path = os.path.join(test_images_path, image_file)
        image_np = plt.imread(image_path)

        detections = detect_objects(image_np, detection_model, multiple_detection)

        if detections is not None:
            save_detections(image_np, detections, category_index, image_file, multiple_detection)
        else:
            print(f"No objects detected in {image_file}.")

    print("Confusion Matrix:")
    print(confusion_matrix)

    iou_thresholds = [0.5]

    evaluation_results = []

    for class_id in range(1, num_classes + 1):
        class_name = category_index[class_id]['name']
        class_precision = precision[class_id][0]
        class_recall = recall[class_id][0]
        class_f1_score = f1_score[class_id][0]
        class_mAP = mean_average_precision[class_name]
        class_iou = iou[class_id][iou_thresholds.index(iou_thresholds[0])]
        class_acc = class_accuracy[class_id]  # Retrieve class accuracy

        evaluation_results.append({
            'Class': class_name,
            'Accuracy': class_acc,
            'Precision': class_precision,
            'Recall': class_recall,
            'F1-Score': class_f1_score,
            'mAP': class_mAP,
            f'IoU @ {iou_thresholds[0]}': class_iou
        })

    evaluation_results.append({
        'Class': 'Overall Accuracy',
        'Accuracy': overall_accuracy,
        'Precision': '',
        'Recall': '',
        'F1-Score': '',
        'mAP': '',
        f'IoU @ {iou_thresholds[0]}': ''
    })

    df = pd.DataFrame(evaluation_results)

    columns_order = ['Class', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'mAP',
                     f'IoU @ {iou_thresholds[0]}']
    df = df[columns_order]

    print(df)

    df.to_csv('evaluation_results.csv', index=False)

    output_folder = 'graphs'
    for class_id in range(1, 6):
        class_name = category_index[class_id]['name']
        plt.figure()
        plt.step(recall[class_id], precision[class_id], where='post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for Class - {class_name}')
        output_path = os.path.join(output_folder, f'precision_recall_class_{class_name}.png')
        plt.savefig(output_path)
        plt.close()

    class_names = [category_index[i]['name'] for i in range(1, num_classes + 1)]
    output_path = os.path.join(output_folder, f'confusion_matrix.png')
    plot_confusion_matrix(confusion_matrix, class_names, output_path)

# def main():
#     detection_model = tf.saved_model.load(model_path)
#     label_map = label_map_util.load_labelmap(label_map_path)
#     categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=5, use_display_name=True)
#     category_index = label_map_util.create_category_index(categories)
#
#     image_path = os.path.join(test_images_path, input_image)
#     image_np = plt.imread(image_path)
#
#     multiple_detection = False

#     detections = detect_objects(image_np, detection_model, multiple_detection)
#
#     save_detections(image_np, detections, category_index, image_file, multiple_detection=True)

if __name__ == '__main__':
    main(multiple_detection)

