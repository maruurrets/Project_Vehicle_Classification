from utils.data_aug import create_data_aug_layer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers

def create_model(
    weights: str = "imagenet",
    input_shape: tuple = (224, 224, 3),
    dropout_rate: float = 0.0,
    data_aug_layer: dict = None,
    classes: int = None,
):
    
    """
    Creates and loads the Resnet50 model we will use for our experiments.
    Depending on the `weights` parameter, this function will return one of
    two possible keras models:
        1. weights='imagenet': Returns a model ready for performing finetuning
                               on your custom dataset using imagenet weights
                               as starting point.
        2. weights!='imagenet': Then `weights` must be a valid path to a
                                pre-trained model on our custom dataset.
                                This function will return a model that can
                                be used to get predictions on our custom task.

    See an extensive tutorial about finetuning with Keras here:
    https://www.tensorflow.org/tutorials/images/transfer_learning.

    Parameters
    ----------
    weights : str
        One of None (random initialization),
        'imagenet' (pre-training on ImageNet), or the path to the
        weights file to be loaded.

    input_shape	: tuple
        Model input image shape as (height, width, channels).
        Only needed when weights='imagenet'. Otherwise, the trained model
        already has the input shape defined and we shouldn't change it.
        Input image size cannot be no smaller than 32. E.g. (224, 224, 3)
        would be one valid value.

    dropout_rate : float
        Value used for Dropout layer to randomly set input units
        to 0 with a frequency of `dropout_rate` at each step during training
        time, which helps prevent overfitting.
        Only needed when weights='imagenet'.

    data_aug_layer : dict
        Configuration from experiment YAML file used to setup the data
        augmentation process during finetuning.
        Only needed when weights='imagenet'.

    classes : int
        Model output classes.
        Only needed when weights='imagenet'. Otherwise, the trained model
        already has the output classes number defined and we shouldn't change
        it.

    Returns
    -------
    model : keras.Model
        Loaded model either ready for performing finetuning or to start doing
        predictions.
    """

    # Create the model to be used for finetuning here!
    if weights == "imagenet":
        # Define the Input layer
        # Assign it to `input` variable
        # Use keras.layers.Input(), following this requirements:
        #   1. layer dtype must be tensorflow.float32
        # TODO
        input = None
        input = tf.keras.layers.Input(dtype=tf.float32, shape=input_shape)
        
        
        # Create the data augmentation layers here and add to the model next
        # to the input layer
        # If no data augmentation was used, skip this
        # TODO
        if data_aug_layer:
            data_augmentation = create_data_aug_layer(data_aug_layer)
            x = data_augmentation(input)
            x = keras.applications.resnet50.preprocess_input(x)
        else:
            x = keras.applications.resnet50.preprocess_input(input)
        
        # Add a layer for preprocessing the input images values
        # E.g. change pixels interval from [0, 255] to [0, 1]
        # Resnet50 already has a preprocessing function you must use here
        # See keras.applications.resnet50.preprocess_input()
        # TODO
        #x = tf.keras.layers.Rescaling(scale=1./255)(x)

        # Create the corresponding core model using
        # keras.applications.ResNet50()
        # The model created here must follow this requirements:
        #   1. Use imagenet weights
        #   2. Drop top layer (imagenet classification layer)
        #   3. Use Global average pooling as model output
        # TODO
        #tf.keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000, **kwargs)
        base_model = keras.applications.resnet50.ResNet50(include_top=False, weights=weights, pooling='avg', input_shape=input_shape)
        
        #unfreezing the last 30 layers
        for layer in base_model.layers[-30:]:
            layer.trainable = True
        
        base_model.trainable = False

        x=base_model(x, training=False)
        
        # Add a single dropout layer for regularization, use
        # keras.layers.Dropout()
        # TODO
        x = tf.keras.layers.Dropout(dropout_rate)(x)

        # Add the classification layer here, use keras.layers.Dense() and
        # `classes` parameter
        # Assign it to `outputs` variable
        # TODO
        #outputs = None
        x2 = keras.layers.Dense(classes, kernel_regularizer=regularizers.l2(0.0005), activation='softmax')
        outputs = x2(x)

        # Now you have all the layers in place, create a new model
        # Use keras.Model()
        # Assign it to `model` variable
        # TODO
        #model = None
        model = keras.Model(input, outputs)
    else:
        # For this particular case we want to load our already defined and
        # finetuned model, see how to do this using keras
        # Assign it to `model` variable
        # TODO
        model = None
        model=keras.models.load_model(weights) 

    return model
