import cv2
import numpy as np
import sys
import getopt
import time
import os

try:
    SHOW_IMAGE = True
    import tensorflow.lite as tflite
except ImportError:
    SHOW_IMAGE = False
    import tflite_runtime.interpreter as tflite

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240


def load_labels(label_path):
    with open(label_path) as f:
        labels = {}
        index = 0
        for line in f.readlines():
            labels[index] = line.rstrip("\n")
            index = index + 1
        return labels


def load_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path, num_threads=4)
    interpreter.allocate_tensors()
    return interpreter


def process_image(interpreter, image, input_index):
    input_data = np.expand_dims(image, axis=0)

    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    output_details = interpreter.get_output_details()

    positions = np.squeeze(interpreter.get_tensor(output_details[0]["index"]))
    classes = np.squeeze(interpreter.get_tensor(output_details[1]["index"]))
    scores = np.squeeze(interpreter.get_tensor(output_details[2]["index"]))

    result = []

    for idx, score in enumerate(scores):
        if score > 0.5:
            result.append({"pos": positions[idx], "_id": classes[idx]})

    return result


def display_result(result, frame, labels):
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 0.6
    color = (255, 0, 0)
    thickness = 1

    r = 640.0 / frame.shape[1]
    dim = (640, int(frame.shape[0] * r))
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    width = frame.shape[1]
    height = frame.shape[0]

    for obj in result:
        pos = obj["pos"]
        _id = obj["_id"]

        x1 = int(pos[1] * width)
        x2 = int(pos[3] * width)
        y1 = int(pos[0] * height)
        y2 = int(pos[2] * height)

        cv2.putText(frame, labels[_id], (x1, y1), font, size, color, thickness)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    cv2.imshow("Object Detection", frame)


def main(argv):
    dir_path = None

    try:
        opts, _ = getopt.getopt(argv, "hd:", ["help", "directory="])
        for opt, arg in opts:
            if opt == "-h" or opt == "--help":
                raise Exception()
            elif opt in ("-d", "--directory"):
                dir_path = arg

        if not dir_path:
            raise Exception("Please specify a directory using -d or --directory option.")

        if not os.path.isdir(dir_path):
            raise Exception("Directory '{}' does not exist.".format(dir_path))

    except Exception as e:
        print("Error:", e)
        print("Usage: test-camera.py -d <directory>")
        sys.exit(2)

    model_path = os.path.join(dir_path, "model.tflite")
    label_path = os.path.join(dir_path, "labels.txt")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow as a backup if MSMF fails
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 15)

    interpreter = load_model(model_path)
    labels = load_labels(label_path)

    input_details = interpreter.get_input_details()
    input_shape = input_details[0]["shape"]
    height = input_shape[1]
    width = input_shape[2]
    input_index = input_details[0]["index"]

    while True:
        ret, frame = cap.read()

        if ret:
            start = round(time.time() * 1000)
            image = cv2.resize(frame, (width, height))
            stop = round(time.time() * 1000)
            print("-------TIMING--------")
            print("resize image: {} ms".format(stop - start))

            start = round(time.time() * 1000)
            top_result = process_image(interpreter, image, input_index)
            stop = round(time.time() * 1000)
            print("process image: {} ms".format(stop - start))
            print("-------RESULTS--------")

            for obj in top_result:
                pos = obj["pos"]
                _id = obj["_id"]
                x1 = int(pos[1] * CAMERA_WIDTH)
                x2 = int(pos[3] * CAMERA_WIDTH)
                y1 = int(pos[0] * CAMERA_HEIGHT)
                y2 = int(pos[2] * CAMERA_HEIGHT)
                print("class: {}, x1: {}, y1: {}, x2: {}, y2: {}".format(labels[_id], x1, y1, x2, y2))

            if SHOW_IMAGE:
                display_result(top_result, frame, labels)
                c = cv2.waitKey(1)
                if c == 27:
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1:])
