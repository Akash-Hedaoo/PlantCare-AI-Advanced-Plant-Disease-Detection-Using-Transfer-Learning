# 2_Model_Building/mobilenetv2_base.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Input

def load_base(input_shape=(224,224,3), weights="imagenet"):
    """
    Load MobileNetV2 base (without top).
    Returns the base model (not trainable by default).
    """
    base_model = MobileNetV2(include_top=False, weights=weights, input_shape=input_shape)
    base_model.trainable = False  # freeze base
    return base_model

if __name__ == "__main__":
    base = load_base()
    print("MobileNetV2 base summary (frozen):")
    base.summary()