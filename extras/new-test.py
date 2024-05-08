import tensorflow as tf
import numpy as np
import cv2

def process_image(interpreter, image, input_index):
    input_data = np.expand_dims(image, axis=0)

    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    classes = np.squeeze(interpreter.get_tensor(output_details[-1]["index"]))
    scores = np.squeeze(interpreter.get_tensor(output_details[0]["index"]))
    positions = np.squeeze(interpreter.get_tensor(output_details[1]["index"]))

    result = []

    for idx, score in enumerate(scores):
        if score > 0.5:
            result.append({"pos": positions[idx], "_id": classes[idx] })

    return result

def main(image_path, model_path, label_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image = np.zeros((240, 320, 3), dtype=np.uint8)
    preprocessed_image = preprocess_image(image)

    result = process_image(interpreter, preprocessed_image, input_details[0]['index'])
    print(result)

def preprocess_image(image):
    resized_image = cv2.resize(image, (320, 320))
    return resized_image

if __name__ == "__main__":
    image_path = "C:/Users/aadit/Desktop/ml/robo/machine-learning-main/test/white-tab/cam320x240-BOHOEL-1665095395786.jpeg"
    model_path = "C:/Users/aadit/Desktop/ml/robo/machine-learning-main/object-detection/build/sorting_line/model.tflite"
    label_path = "C:/Users/aadit/Desktop/ml/robo/machine-learning-main/object-detection/build/sorting_line/labels.txt"

    main(image_path, model_path, label_path)
