import cv2
import numpy as np
import sys, getopt,time

try:
    SHOW_IMAGE = True
    import tensorflow.lite as tflite
except ImportError:
    SHOW_IMAGE = False
    import tflite_runtime.interpreter as tflite

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240


def load_labels(label_path):
    with open(label_path, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def process_image(interpreter, image, input_index, k=3):
    input_data = np.expand_dims(image, axis=0)

    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    output_data = np.squeeze(output_data)

    top_k = output_data.argsort()[-k:][::-1]
    result = []
    for _id in top_k:
        score = float(output_data[_id] / 255.0)
        result.append((_id, score))

    return result


def display_result(top_result, frame, labels):
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 0.6
    color = (255, 0, 0)
    thickness = 1

    r = 640.0 / frame.shape[1]
    dim = (640, int(frame.shape[0] * r))
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    for idx, (_id, score) in enumerate(top_result):
        x = 12
        y = 24 * idx + 24
        cv2.putText(frame, "{} - {:0.4f}".format(labels[_id], score),
                    (x, y), font, size, color, thickness)

    cv2.imshow("Image Classification", frame)


def main(argv):
    dir = None

    try:
        opts, _ = getopt.getopt(argv, "hd:", ["help", "directory="])
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                raise ValueError("Help message requested.")
            elif opt in ("-d", "--directory"):
                dir = str(arg)

        if dir is None:
            raise ValueError("No directory provided.")

    except ValueError as e:
        print(e)
        print("Specify a directory in which the model is located.")
        print("test-camera.py -d <directory>")
        sys.exit(2)

    model_path = dir + "/model.tflite"
    label_path = dir + "/labels.txt"

    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Using CAP_DSHOW flag
        if not cap.isOpened():
            raise IOError("Cannot open webcam.")

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
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                stop = round(time.time() * 1000)
                print("-------TIMING--------")
                print("resize image: {} ms".format(stop - start))

                start = round(time.time() * 1000)
                top_result = process_image(interpreter, image, input_index)
                stop = round(time.time() * 1000)

                print("process image: {} ms".format(stop - start))
                print("-------RESULTS--------")
                for idx, (i, score) in enumerate(top_result):
                    print("{} - {:0.4f}".format(labels[i], score))

                if SHOW_IMAGE:
                    display_result(top_result, frame, labels)

                    c = cv2.waitKey(1)
                    if c == 27:
                        break

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting...")
    except Exception as e:
        print("An error occurred:", e)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1:])
