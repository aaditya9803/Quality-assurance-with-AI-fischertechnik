import cv2
import numpy as np
import sys
import getopt
import time
import tensorflow as tf

def load_labels(label_path):
    with open(label_path) as f:
        labels = {}
        index = 0
        for line in f.readlines():
            labels[index] = line.rstrip("\n")
            index = index + 1
        return labels

def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def process_image(interpreter, image, input_index, confidence_threshold=0.3):
    input_data = np.expand_dims(image, axis=0)
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    num_boxes = min(boxes.shape[1], scores.shape[0])

    for i in range(num_boxes):
        score = scores[i]
        if score > confidence_threshold:
            ymin, xmin, ymax, xmax = boxes[0, i]

            xmin = int(xmin * image.shape[1])
            xmax = int(xmax * image.shape[1])
            ymin = int(ymin * image.shape[0])
            ymax = int(ymax * image.shape[0])

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            class_index = int(classes[i])
            class_label = labels[class_index]

            label = "{}: {:.2f}".format(class_label, score)
            cv2.putText(image, label, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Object Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def evaluate_model(model_dir, test_image_path):
    model_path = model_dir + "/model.tflite"
    label_path = model_dir + "/labels.txt"

    interpreter = load_model(model_path)
    labels = load_labels(label_path)

    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']

    height = input_shape[1]
    width = input_shape[2]

    frame = cv2.imread(test_image_path, cv2.IMREAD_COLOR)
    if frame is None:
        print("Error loading image:", test_image_path)
        return

    image = cv2.resize(frame, (width, height))
    if image is None:
        print("Error resizing image:", test_image_path)
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    input_index = input_details[0]["index"]
    process_image(interpreter, image, input_index)

    cv2.imshow("Object Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(argv):
    dir = None
    test_image_path = None

    try:
        opts, _ = getopt.getopt(argv, "hd:i:", ["help", "directory=", "image="])
        for opt, arg in opts:
            if opt == "-h":
                raise Exception()
            elif opt in ("-d", "--directory"):
                dir = str(arg)
            elif opt in ("-i", "--image"):
                test_image_path = str(arg)

        if dir is None or test_image_path is None:
            raise Exception()

    except Exception:
        print("Specify a directory where the model is located")
        print("and a path to the test image.")
        print("test-image.py -d <directory> -i <image_path>")
        sys.exit(2)

    evaluate_model(dir, test_image_path)

if __name__ == "__main__":
    main(sys.argv[1:])
