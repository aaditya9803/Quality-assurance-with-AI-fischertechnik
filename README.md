# TensorFlow example project

## Python packages for creating models on a 64 bit device
* pip install -q tensorflow==2.5.0
* pip install -q --use-deprecated=legacy-resolver tflite-model-maker
* pip install -q pycocotools

## macOS Montery (12.4) Python 3.9.13
* pip install numpy==1.20.3 
* pip install tensorflow==2.5.0
* pip install tflite_model_maker==0.3.2
* pip install pycocotools==2.0.2

## Python packages for the TXT 4 controller (32 bit)
* pip install --index-url https://google-coral.github.io/py-repo/ tflite_runtime

## Image detection
* go to image-detection directory
* create model with:

  ``python create-model.py -d <path from source directory>``

* test model with:

  ``python test-image.py -d <model path> -i <image>``

  ``python test-camera.py -d <model path>``

## Object detection
* go to object-detection directory
* create model with:

  ``python create-model.py -d <path from source directory>``

* test model with:

  ``python test-image.py -d <model path> -i <image>``

  ``python test-camera.py -d <model path>``

## Links
* https://www.tensorflow.org/lite/tutorials/model_maker_image_classification
* https://www.tensorflow.org/lite/tutorials/model_maker_object_detection
