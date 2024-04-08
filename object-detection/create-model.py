# https://www.tensorflow.org/lite/tutorials/model_maker_object_detection
# https://github.com/tzutalin/labelImg

import sys, getopt, time, pathlib
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

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

    # select object recognition model architecture
    spec = model_spec.get("efficientdet_lite0")
    spec.config.var_freeze_expr = "(efficientnet|fpn_cells|resample_p6)"
    spec.config.tflite_max_detections = 25

    print(spec.config)

    # load input data specific to an on-device ML app
    train_data, validation_data, test_data = object_detector.DataLoader.from_csv(dir + "/dataset.csv")

    # customize the TensorFlow model
    model = object_detector.create(
        train_data,
        model_spec=spec,
        batch_size=batch_size,
        epochs=epochs,
        train_whole_model=False
    )
    model.summary()

    # evaluate the model
    print(model.evaluate(test_data))

    # export to Tensorflow Lite model and label file in `export_dir`
    path = pathlib.PurePath(dir)
    model.export(export_dir="build/" + path.name + "/")
    model.export(export_dir="build/" + path.name + "/", export_format=ExportFormat.LABEL)

    # evaluate the tensorflow lite model
    print(model.evaluate_tflite("build/" + path.name + "/model.tflite", test_data))

    stop = round(time.time() * 1000)
    print("process image: {} ms".format(stop - start))

if __name__ == "__main__":
   main(sys.argv[1:])