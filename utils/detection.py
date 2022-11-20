from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

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
    variable) over the image, get the vehicle position in the picture
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
    classes = outputs["instances"].pred_classes.cpu().numpy()
    boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

    #classes = outputs["instances"].predict_classes().numpy()
    #boxes = outputs["instances"].predict_boxes.tensor().numpy()

    if len(classes) != 0 and len(boxes) !=0:
        index = -1
        box_area_list = []
        coordinates_list = []
        for class_num in classes:
            index += 1
            if class_num == 2 or class_num == 7:
                #[left, top, right, bottom]
                x1 = int(boxes[index][0])
                y1 = int(boxes[index][1])
                x2 = int(boxes[index][2])
                y2 = int(boxes[index][3])
                box_area_list.append(abs(x2-x1)* abs(y2-y1))
                coordinates_list.append((x1,y1,x2,y2))
                max_area_index = box_area_list.index(max(box_area_list))
                box_coordinates = coordinates_list[max_area_index]
            else:
                h = img.shape[0]
                w = img.shape[1]
                box_coordinates = [0,0,w,h]
    else:
        h = img.shape[0]
        w = img.shape[1]
        #coordinates that cover the full image, i.e. [0, 0, width, height]
        box_coordinates = [0,0,w,h]

    return box_coordinates
