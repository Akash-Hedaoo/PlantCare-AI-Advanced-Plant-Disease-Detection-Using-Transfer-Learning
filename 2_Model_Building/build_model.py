# 2_Model_Building/build_model.py
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from config import IMG_SIZE, LEARNING_RATE, DROPOUT_RATE
from utils import get_num_classes

def build_model(num_classes, input_shape=(224,224,3), dropout_rate=DROPOUT_RATE, base_trainable=False):
    # Load base
    base = MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape)
    base.trainable = base_trainable  # False -> freeze, True -> train entire base (not recommended)
    
    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)         # keep base in inference mode for BN layers if frozen
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name="mobilenetv2_plant_disease_classifier")
    
    opt = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def unfreeze_last_layers(model, base_name="mobilenetv2_1.00_224", num_layers=20):
    """
    Unfreeze the last `num_layers` layers of the base model for fine-tuning.
    Call this after initial training (and maybe lowering lr).
    """
    # find base model inside model.layers
    for layer in model.layers:
        if layer.name.startswith("mobilenetv2"):
            base = layer
            break
    else:
        raise ValueError("Base model not found by name filter.")
    
    # unfreeze last num_layers layers
    total = len(base.layers)
    for l in base.layers[-num_layers:]:
        l.trainable = True

if __name__ == "__main__":
    import config
    n_classes, _, _ = get_num_classes(config.TRAIN_DIR)
    print("Building model for", n_classes, "classes")
    model = build_model(n_classes, input_shape=(*config.IMG_SIZE, 3))
    model.summary()