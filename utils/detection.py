from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# The chosen detector model is "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
# because this particular model has a good balance between accuracy and speed.
# You can check the following Colab notebook with examples on how to run
# Detectron2 models
# https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5.
# Assign the loaded detection model to global variable DET_MODEL

#create a detectron2 config and a detectron2 DefaultPredictor to run inference on this images
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
)
try:
    # It may fail if no GPU was found
    DET_MODEL = DefaultPredictor(cfg)
except:
    # Load the model for CPU only
    print(
        f"Failed to load Detection model on GPU, "
        "trying with CPU. Message: {exp}."
    )
    cfg.MODEL.DEVICE='cpu'
    DET_MODEL = DefaultPredictor(cfg)


def get_vehicle_coordinates(img):
    """
    This function will run an object detector (loaded in DET_MODEL model
    variable) over the the image, get the vehicle position in the picture
    and return it.

    Many things should be taken into account to make it work:
        1. Current model being used can detect up to 80 different objects,
           we're only looking for 'cars' or 'trucks', so you should ignore
           other detected objects.
        2. The object detector may find more than one vehicle in the picture,
           you must then, choose the one with the largest area in the image.
        3. The model can also fail and detect zero objects in the picture,
           in that case, you should return coordinates that cover the full
           image, i.e. [0, 0, width, height].
        4. Coordinates values must be integers, we're making reference to
           a position in a numpy.array, we can't use float values.

    Parameters
    ----------
    img : numpy.ndarray
        Image in RGB format.

    Returns
    -------
    box_coordinates : list
        List having bounding box coordinates as [left, top, right, bottom].
        Also known as [x1, y1, x2, y2].
    """
    # TODO
    box_coordinates = None

    outputs = DET_MODEL(img)

    interest_classes=[2,7]
    classes=outputs["instances"].pred_classes.cpu().numpy()
    boxes=outputs["instances"].pred_boxes.tensor.cpu().numpy()
    print(classes)
    print(boxes)

    
 #select boxes of interest_clases
    n=classes.shape[0]
    interest_boxes=[]
    for i in range(n):
        if classes[i] in interest_classes:
            interest_boxes.append(boxes[i,:])
    
    interest_images=[]
    for bbox in interest_boxes:
        x0,y0,x1,y1=bbox.astype(int)
        area=(y1-y0)*(x1-x0)
        
        #print(x0,y0,x1,y1)
        interest_image= img[y0:y1, x0:x1,:]
        interest_images.append(interest_image)

    return box_coordinates
