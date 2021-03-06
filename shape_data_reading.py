# shape_data_reading.py
import os
import sys
import random
import math
import re
import time
import numpy as np
# import cv2
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from skimage.draw import circle, polygon
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from config import Config
import utils
import model as modellib
import visualize
from model import log


# %matplotlib inline 

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Data Path
#TRAIN_PATH = 'data/stage1_train/'
#TEST_PATH = 'data/stage1_test/'
#TRAIN_PATH2 = 'data/stage1_train2/'
#TRAIN_PATH2 = TRAIN_PATH
#TRAIN_PATH = 'data/stage1_train_color/'
#TEST_PATH = 'data/test_stage1_train_color/'
#TRAIN_PATH = 'data/stage1_train_color/'
#TEST_PATH = 'data/test_stage1_train_color/'

TRAIN_PATH = 'data/stage1_train/'
TEST_PATH = 'data/stage2_test_final/'

TRAIN_PATH2 = TRAIN_PATH


# Get train and test IDs
# train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

IMAGE_DIR = TEST_PATH

# =================== 
class ShapesConfig(Config):
    """Configuration for training on the nuclei dataset.
    Derives from the base Config class and overrides values specific
    to the nuclei dataset.
    """
    # Give the configuration a recognizable name
    NAME = "nuclei"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # If True, pad images with zeros such that they're (max_dim by max_dim)
    IMAGE_PADDING = True  # currently, the False option is not supported

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    RPN_TRAIN_ANCHORS_PER_IMAGE = 320

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 2000
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]
    TRAIN_ROIS_PER_IMAGE = 512
    RPN_NMS_THRESHOLD = 0.7

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 256

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 400

    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7 # may be smaller?

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 512

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 300

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 70
    
config = ShapesConfig()
config.display()

## =================
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

## =================
class ShapesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    Extend the Dataset class and add a method to load the shapes dataset, 
    load_shapes(), and override the following methods:
    load_image()
    load_mask()
    image_reference()
    """

    def load_shapes(self,PATH):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        train_ids = next(os.walk(PATH))[1]
        count = len(next(os.walk(PATH))[1])
        # Add classes
        self.add_class("nuclei", 1, "nucleu")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). 
        for i in range(count):
            id_ = train_ids[i]
            path_i = PATH + id_+ '/images/' + id_ + '.png'
            self.add_image("nuclei", image_id=i, path=path_i)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = imread(self.image_info[image_id]['path'])[:,:,0:3]
        image = resize(image, (256, 256), mode='constant', preserve_range = True)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nuclei":
            return info["nuclei"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        import_path=self.image_info[image_id]['path'].split('/')[1]
        if import_path !="stage1_test":
            # info = self.image_info[image_id]
            PATH = self.image_info[image_id]['path'].split('/')[0] +"/" + self.image_info[image_id]['path'].split('/')[1] + "/"
            # print("load_mask_PATH:",PATH)
            train_ids = next(os.walk(PATH))[1]
            id_ = train_ids[image_id]
            path = PATH + id_
            count = len(next(os.walk(path + '/masks/'))[2])
            mask = np.zeros([256, 256, count], dtype=np.uint8)
            # mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
            i = 0
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                if mask_file[0] == ".":
                    mask_file=mask_file[2:]
            #    if "." in mask_file[0]:
            #        print("full path of nuc:", path + '/masks/' + mask_file)
                mask_ = imread(path + '/masks/' + mask_file)
#                if len(mask_.shape==3):
#                    mask_=mask_[:,:,0]
                mask_ = np.expand_dims(resize(mask_, (256, 256), mode='constant', 
                                            preserve_range=True), axis=-1)
                mask[:, :, i:i+1] = mask_
                # mask = np.maximum(mask, mask_)
                i += 1

            # Handle occlusions
            # each pixel in each 1st 2ed dim
            occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
            # 1 is blank in occlusion, 0 is image
            for i in range(count-2, -1, -1):
                mask[:, :, i] = mask[:, :, i] * occlusion
                occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
            # Map class names to class IDs.
            class_ids = np.array([1 for s in range(count)])
            return mask, class_ids.astype(np.int32)
        else:
            pass


## =================
# Training dataset
dataset_train = ShapesDataset()
dataset_train.load_shapes(TRAIN_PATH)
dataset_train.prepare()

# Validation dataset
dataset_val = ShapesDataset()
dataset_val.load_shapes(TRAIN_PATH2)
dataset_val.prepare()

# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


## ================ Create model ================ ##

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "last"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)


## ================ train model ================ ##
# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=20, 
            layers='heads')

# # Train the layers from ResNet stage 4 and up
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=40, 
            layers='4+')

# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE,
            epochs=60, 
            layers="all")

# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)


## ================ Detection ================ ##
class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config, 
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                            dataset_train.class_names, figsize=(8, 8))


## ================ Evaluation ================ ##
# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 4)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)
    
print("mAP: ", np.mean(APs))


# ## ================ Prediction ================ ##
# Predict dataset
dataset_predict = ShapesDataset()
dataset_predict.load_shapes(TEST_PATH)
dataset_predict.prepare()
print(dataset_predict.image_ids)
import gc

def Handle_occlusions(mask):
    # Handle occlusions
    # each pixel in each 1st 2ed dim
    count = mask.shape[2]
    occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
    # 1 is blank in occlusion, 0 is image
    for i in range(count-2, -1, -1):
        mask[:, :, i] = mask[:, :, i] * occlusion
        occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
    return mask

# get the size of test_image
shape_test_image = []
for id_ in next(os.walk(TEST_PATH))[1]:
    img = imread(TEST_PATH + id_ + '/images/' + id_ + '.png')
    shape_test_image.append(img.shape)

pred_result = []
for i in range(len(dataset_predict.image_ids)):
    # print(i)
    # image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    #     modellib.load_image_gt(dataset_predict, inference_config, i, use_mini_mask=False)
    image = dataset_predict.load_image(i)
    shape_0 = shape_test_image[i][0]
    shape_1 = shape_test_image[i][1]
    results = model.detect([image], verbose=0)
    r = results[0]
    #print("r['masks']:",r['masks'].shape)
    if len(r['masks']) >0:
    	mask_re = resize(r['masks'], (shape_0, shape_1,), mode='constant', preserve_range = True)
    	## delete the repeat pixel
    	mask_re = Handle_occlusions(mask_re)
    	pred_result.append(mask_re)
    #print(mask_re.shape,next(os.walk(TEST_PATH))[1][i])
    #print("total len:",shape_0*shape_1)
    # resize(r['masks'], (256, 256), mode='constant', preserve_range = True)
    gc.collect()

with open ("pred_result_stage2.list","w") as f:
    for i in pred_result:
        for j in i:
            # for k in j:
            f.write(str(j)+"\t")
            # f.write("\n")
    f.write("\n")


# ## ================ run-length encoding ================ ##
# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    print(lab_img.max())
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

# iterate over the test IDs and generate run-length encodings for each seperate mask identified by skimage
new_test_ids = []
rles = []
for n in range(len(dataset_predict.image_ids)):
    id_ = next(os.walk(TEST_PATH))[1][n]
    #print(n,id_)
    rle=[]
    for i_rle in range(pred_result[n].shape[2]):
        rle.extend(list(prob_to_rles(pred_result[n][:,:,i_rle])))
        rle.sort(key=lambda x:x[0])
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018-stage2.csv', index=False)
