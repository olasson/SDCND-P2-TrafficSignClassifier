# **Traffic Sign Classifier** 

*by olasson*

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

*This is a revised version of my Traffic Sign Classifier project.*

## Project overview

The majority of the project code is located in the folder `code`:

* [`augment.py`](https://github.com/olasson/SDCND-P2-TrafficSignClassifier/blob/master/code/augment.py)
* [`misc.py`](https://github.com/olasson/SDCND-P2-TrafficSignClassifier/blob/master/code/misc.py)
* [`io.py`](https://github.com/olasson/SDCND-P2-TrafficSignClassifier/blob/master/code/io.py)
* [`models.py`](https://github.com/olasson/SDCND-P2-TrafficSignClassifier/blob/master/code/models.py)
* [`process.py`](https://github.com/olasson/SDCND-P2-TrafficSignClassifier/blob/master/code/process.py)
* [`plots.py`](https://github.com/olasson/SDCND-P2-TrafficSignClassifier/blob/master/code/plots.py)

The main project script is called [`traffic_sign_classifier.py`](https://github.com/olasson/SDCND-P2-TrafficSignClassifier/blob/master/traffic_sign_classifier.py). It contains the implementation of a very simple command line tool.

The results of model training and predictions are found in:
* `images/`

The images shown in this readme are found in 

* `images/`

## Command line arguments

The following command line arguments are defined:

#### Data

* `--data_meta:` File path to a .csv data file containing sign meta data.

#### Show

* `--show_images:` File path to a pickled (.p) file containing a set of images.
* `--show_distributions:` File path(s) to pickled file(s) containing a label set.
* `--distribution_title:` Title for the distribution plot.
* `--n_max_images:` The maximum number of images in the image plot.
* `--n_max_cols:` The maximum number of columns in the image plot.

#### Config

* `--model_config:` Path to a .json file containing model config.

#### Misc

* `--force_save:` If enabled, permits overwriting existing data.

## Config system

In order to simplify the process of experimenting with different model architectures and parameters, I have implemented a very simple config system. In order to create new datasets and train a model, the user only needs to define a `.json` file in the folder `config/` on the form (example):

    {
        "model_path": "./models/LeNet_01.h5",
        "data_train": "./data/prepared_train_full.p",
        "data_test": "./data/prepared_test.p",
        "data_valid": "./data/prepared_valid.p",
        "mirror_data": 1,
        "transform_data": 1,
        "lrn_rate": 0.001,
        "n_max_epochs": 50,
        "batch_size": 64
    }

After the config file is created run the command

`python traffic_classifier.py --model_config './config/<config_file_name>.json'`

The [`traffic_sign_classifier.py`](https://github.com/olasson/SDCND-P2-TrafficSignClassifier/blob/master/traffic_sign_classifier.py) script will automatically detect if the specified datasets and model exists or not, and create them if needed.

## Data Exploration

Below is a very basic data overview.

| Dataset   |      # Images      |  # Unique Classes |  Shape |
|----------|:-------------:|------:|------:|
| `train.p` |  34799 | 43 | (32,32,3) |
| `valid.p` |  4410 | 43 | (32,32,3) |
| `test.p` |  12630 | 43 | (32,32,3) |

Lets take a look at a random subset of training images.

<p align="center">
  <img width="80%" height="80%" src="https://github.com/olasson/SDCND-P2-TrafficSignClassifier/blob/master/images/readme/train.png">
</p>

*Observation 1:* The images have uneven brightness. This should be corrected for in the pre processing step. 

Lets compare the class distributions (click on image to enlarge). 

<p align="center">
  <img width="80%" height="80%" src="https://github.com/olasson/SDCND-P2-TrafficSignClassifier/blob/master/images/readme/label_distributions.png">
</p>

*Observation 2:* Very uneven training distribution, which could lead to overfitting.

## Data Preparation

This section is concerned with preparing the datasets for use by a model.

### Augmentation

This attempts to counter *Observation 2* in the previous section through artificially creating more training images until an uniform distribution is created. 

Relevant code: `code/augment.py`

#### Mirroring

It is possible to mirror certain classes to imitate others. This is "formalized" in the following mirror map found in `traffic_sign_classifier.py`:

    MIRROR_MAP = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                  -1, 11, 12, 13, -1, 15, -1, 17, 18, 20,
                  19, -1, 22, -1, -1, -1, 26, -1, -1, -1,
                  30, -1, -1, 34, 33, 35, 37, 36, 39, 38,
                  -1, -1, -1]

The mirror map defines a mapping where *`Class i` is mirrored to imitate `Class mirror_map[i]`*. For example, "Turn Left Ahead" is mirrored to imitate "Turn Right Ahead". The mirror map is used by `augment_data_by_mirroring()`.

#### Random transformations

By applying one or more (up to four) random transformations, new images can be created from existing ones. The following random transformations are defined in `code/augment.py`:

* `scale_image()`
* `translate_image()`
* `prespective_transform()`
* `rotate_image()`

All transformations preserve the original image dimensions. One or more of these is applied by `random_transforms()` which in turn is called by `augment_data_by_random_transform()`. It will apply random transformation to each class until a target count for each class is reached. 

<p align="center">
  <img width="80%" height="80%" src="https://github.com/olasson/SDCND-P2-TrafficSignClassifier/blob/master/images/readme/random_transforms.png">
</p>

For this project, I only applied agumentation to `train.p`. Lets take a look at the distribution after augmentation. 

<p align="center">
  <img width="80%" height="80%" src="https://github.com/olasson/SDCND-P2-TrafficSignClassifier/blob/master/images/readme/prepared_label_distributions.png">
</p>
            
#### Pre-processing

This step is concerned with establishing a "minimum quality" of data that is fed to the model.

Relevant code: `code/process.py`

#### Histogram Equalization. 

In order to combat uneven brightness, histogram equalization (CLAHE) is applied by `histogram_equalization()`. 

<p align="center">
  <img width="80%" height="80%" src="https://github.com/olasson/SDCND-P2-TrafficSignClassifier/blob/master/images/readme/preproc_hist_eq.png">
</p>

The brightness of the images in the bottom row is more equal. Hopefully this will allow the model to focus more on the physical features of the signs, and less on the brightness. 

#### Grayscale

In order to lighten the computation load, grayscale conversion is applied by `grayscale()`. 

#### Normalization 

In order to ensure a set range of values the model has to learn, and in turn (hopefully) cause faster optimizer convergence, normalization is applied by `normalize()`.

## Models

This section defines the models used in the project. 

Relevant code: `code/models.py`

### VGG16 Inspired

The first model, called `VGG16_50_16` is inspired by the [VGG16 architecture](https://neurohive.io/en/popular-networks/vgg16/). A model summary follows below: 

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 32, 32, 8)         208       
    _________________________________________________________________
    activation (Activation)      (None, 32, 32, 8)         0         
    _________________________________________________________________
    batch_normalization (BatchNo (None, 32, 32, 8)         32        
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 16, 16, 8)         0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 16, 16, 16)        1168      
    _________________________________________________________________
    activation_1 (Activation)    (None, 16, 16, 16)        0         
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 16, 16, 16)        64        
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 16, 16, 16)        2320      
    _________________________________________________________________
    activation_2 (Activation)    (None, 16, 16, 16)        0         
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 16, 16, 16)        64        
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 8, 8, 16)          0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 8, 8, 32)          4640      
    _________________________________________________________________
    activation_3 (Activation)    (None, 8, 8, 32)          0         
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 8, 8, 32)          128       
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 8, 8, 32)          9248      
    _________________________________________________________________
    activation_4 (Activation)    (None, 8, 8, 32)          0         
    _________________________________________________________________
    batch_normalization_4 (Batch (None, 8, 8, 32)          128       
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 8, 8, 32)          9248      
    _________________________________________________________________
    activation_5 (Activation)    (None, 8, 8, 32)          0         
    _________________________________________________________________
    batch_normalization_5 (Batch (None, 8, 8, 32)          128       
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 4, 4, 32)          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 512)               0         
    _________________________________________________________________
    dense (Dense)                (None, 128)               65664     
    _________________________________________________________________
    activation_6 (Activation)    (None, 128)               0         
    _________________________________________________________________
    batch_normalization_6 (Batch (None, 128)               512       
    _________________________________________________________________
    dropout (Dropout)            (None, 128)               0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 128)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 128)               16512     
    _________________________________________________________________
    activation_7 (Activation)    (None, 128)               0         
    _________________________________________________________________
    batch_normalization_7 (Batch (None, 128)               512       
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 128)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 43)                5547      
    _________________________________________________________________
    activation_8 (Activation)    (None, 43)                0         
    =================================================================
    Total params: 116,123
    Trainable params: 115,339
    Non-trainable params: 784


All activation layers are of type `relu`, except for `activation_8` which is `softmax`. 

### LeNet Inspired

The first model, called `LeNet_50_16` is inspired by [LeNet architecture](https://en.wikipedia.org/wiki/LeNet). A model summary follows below:

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 28, 28, 6)         156       
    _________________________________________________________________
    activation (Activation)      (None, 28, 28, 6)         0         
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 6)         0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 10, 10, 16)        2416      
    _________________________________________________________________
    activation_1 (Activation)    (None, 10, 10, 16)        0         
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 5, 5, 16)          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 400)               0         
    _________________________________________________________________
    dense (Dense)                (None, 120)               48120     
    _________________________________________________________________
    activation_2 (Activation)    (None, 120)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 84)                10164     
    _________________________________________________________________
    activation_3 (Activation)    (None, 84)                0         
    _________________________________________________________________
    dropout (Dropout)            (None, 84)                0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 43)                3655      
    _________________________________________________________________
    activation_4 (Activation)    (None, 43)                0         
    =================================================================
    Total params: 64,511
    Trainable params: 64,511
    Non-trainable params: 0

All activation layers are of type `relu`, except for `activation_4` which is `softmax`. 

### Training

The user has the option to specify a couple of hyperparameters through the model config file namely `lrn_rate`, `batch_size` and `max_epochs`. While the first two are fairly straight forward, `max_epochs` is, as the name implies, not necessarily the number of epochs training will run for. This is due to the Keras callback implemented:
    
    ...
    
    early_stopping = EarlyStopping(monitor = 'val_accuracy', 
                                   patience = 5, min_delta = 0.001, 
                                   mode = 'max', restore_best_weights = True)
    ...

 
In plain English, this callback does the following: *Stop the traning if there has not been at least a `min_delta` improvement in the metric `monitor` for `patience` epochs.* With the specified values: *Stop the traning if there has not been at least a `0.001` improvement in the metric `val_accuracy` for `5` epochs.*

This will prevent the training from running while no "meaningful" improvement in accuracy is achieved. It is worth noting that this in no way guarantees the best training. 

## Results

Below is the results for all models trained. The models have very generic names, but any details can be found in their respective config file.

| Model Name    | Stop Epoch/Max Epoch  | Evaluation Acc. | Validation Loss  |   
| :---        |    :----:   |          ---: |          ---: |
| `LeNet_01`   |    21/50  |  0.9580 |  0.1290 |
| `LeNet_02`   |    17/50  |  0.9278 |  0.3560 |
| `VGG16_01`   |    15/50  |  0.9614 |  0.1070 |
| `VGG16_02`   |    15/50  |  0.9610 |  0.0770 |

Plots and results for all models can be found in the `images/readme` folder. 

The results on random web images are shown below

### LeNet_01

Overall, this model seems very accurate as it correctly identifies all web images. It was trained on the fully agumented dataset. 

<p align="center">
  <img width="80%" height="80%" src="https://github.com/olasson/SDCND-P2-TrafficSignClassifier/blob/master/images/readme/web_set_predictions_by_LeNet_01.png">
</p>

### LeNet_02

This model is less accurate than its first iteration. The dataset for this model was mirrored only, which has a very uneven distribution of sign labels. The cause of the incorrect predictions might be overfitting. 

<p align="center">
  <img width="80%" height="80%" src="https://github.com/olasson/SDCND-P2-TrafficSignClassifier/blob/master/images/readme/web_set_predictions_by_model_LeNet_02.png">
</p>


### VGG16_01

Overall, this model seems very accurate as it correctly identifies all web images. It was trained on the fully agumented dataset. 

<p align="center">
  <img width="80%" height="80%" src="https://github.com/olasson/SDCND-P2-TrafficSignClassifier/blob/master/images/readme/web_set_predictions_by_VGG16_01.png">
</p>


### VGG16_02

This model is also less accurate than its first iteration. The dataset for this model was random transforms only. This dataset will not expose the model to the same "variation" of images as the fully augmented dataset.

<p align="center">
  <img width="80%" height="80%" src="https://github.com/olasson/SDCND-P2-TrafficSignClassifier/blob/master/images/readme/web_set_predictions_by_VGG16_02.png">
</p>

