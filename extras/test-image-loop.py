import cv2
import numpy as np
import sys
import getopt
import os

try:
    SHOW_IMAGE = True
    import tensorflow.lite as tflite
except ImportError:
    SHOW_IMAGE = False
    import tflite_runtime.interpreter as tflite


def load_labels(label_path):
    """Returns a list of labels"""
    with open(label_path) as f:
        labels = {}
        index = 0
        for line in f.readlines():
            labels[index] = line.rstrip("\n")
            index = index + 1
        return labels


def load_model(model_path):
    """Load TFLite model, returns a Interpreter instance."""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def process_image(interpreter, image, input_index):
    """Process an image, Return a list of detected class ids and positions"""
    input_data = np.expand_dims(image, axis=0) 
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    classes = np.squeeze(interpreter.get_tensor(output_details[-1]["index"]))
    scores = np.squeeze(interpreter.get_tensor(output_details[0]["index"]))
    positions = np.squeeze(interpreter.get_tensor(output_details[1]["index"]))

    result = []

    for idx, score in enumerate(scores):
        if score >= 0.2:
            result.append({"pos": positions[idx], "_id": classes[idx]})

    return result


def evaluate_model(model_path, label_path, test_image_dir):
    interpreter = load_model(model_path)
    labels = load_labels(label_path)

    correct_predictions = 0
    total_predictions = 0

    for image_file in os.listdir(test_image_dir):
        if not image_file.endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(test_image_dir, image_file)

        frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if frame is None:
            print("Error loading image:", image_path)
            continue

        input_details = interpreter.get_input_details()
        input_shape = input_details[0]["shape"]
        height, width = input_shape[1], input_shape[2]

        image = cv2.resize(frame, (width, height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_index = input_details[0]["index"]
        top_result = process_image(interpreter, image, input_index)

        for obj in top_result:
            predicted_class = obj["_id"]
            correct_predictions += 1 
            total_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print("Accuracy:", accuracy)


def main(argv):

    dir = None
    test_image_dir = None

    try:

        opts, _ = getopt.getopt(argv, "hd:t:", ["help", "directory=", "test="])
        for opt, arg in opts:
            if opt == "-h, --help":
                raise Exception()
            elif opt in ("-d", "--directory"):
                dir = str(arg)
            elif opt in ("-t", "--test"):
                test_image_dir = str(arg)

        if dir is None or test_image_dir is None:
            raise Exception()

    except Exception:
        print("Specify a directory in which the model is located")
        print("and a directory containing test images.")
        print("test-image.py -d <directory> -t <test_image_dir>")
        sys.exit(2)

    model_path = os.path.join(dir, "model.tflite")
    label_path = os.path.join(dir, "labels.txt")

    evaluate_model(model_path, label_path, test_image_dir)


if __name__ == "__main__":
    main(sys.argv[1:])