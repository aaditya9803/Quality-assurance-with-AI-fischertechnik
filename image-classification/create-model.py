# https://www.tensorflow.org/lite/tutorials/model_maker_image_classification

import sys, getopt, time, pathlib
from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader
from tflite_model_maker.config import ExportFormat

def main(argv):
    
    dir = None
    batch_size = 8
    epochs = 50

    try:

        opts, _ = getopt.getopt(argv, "hd:b:e:", ["help","directory=","batchSize=","epochs="])
        for opt, arg in opts:
            print(opt + ":" + arg)
            if opt == "-h, --help":
                raise Exception()
            elif opt in ("-d", "--directory"):
                dir = str(arg)
            elif opt in ("-b", "--batchSize"):
                batch_size = int(arg)
            elif opt in ("-e", "--epochs"):
                epochs = int(arg)
        
        if (dir is None):
            raise Exception()
    
    except Exception:
        print("Specify a directory that contains your dataset.")
        print("create-model.py -d <directory of dataset>")
        sys.exit(2)

    start = round(time.time() * 1000)

    # load input data specific to an on-device ML app
    data = DataLoader.from_folder(dir + "/")
    train_data, rest_data = data.split(0.8)
    validation_data, test_data = rest_data.split(0.5)

    # select object recognition model architecture
    spec = model_spec.get("mobilenet_v2")

    # customize the TensorFlow model
    model = image_classifier.create(
        train_data,
        model_spec=spec,
        batch_size=batch_size,
        epochs=epochs,
        train_whole_model=True,
        validation_data=validation_data
    )
    model.summary()

    # evaluate the model
    result = model.evaluate(test_data)
    print("test loss, test acc:", result)

    # export to Tensorflow Lite model and label file in `export_dir`
    path = pathlib.PurePath(dir)
    model.export(export_dir="build/" + path.name + "/")
    model.export(export_dir="build/" + path.name + "/", export_format=ExportFormat.LABEL)

    # evaluate the tensorflow lite model
    result = model.evaluate_tflite("build/" + path.name + "/model.tflite", test_data)
    print("test loss, test acc:", result)

    stop = round(time.time() * 1000)
    print("process image: {} ms".format(stop - start))
   
   
if __name__ == "__main__":
   main(sys.argv[1:])

