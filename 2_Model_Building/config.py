import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DIR = os.path.join(BASE_DIR, "1_Data_Collection_and_Preprocessing", "train")
VALID_DIR = os.path.join(BASE_DIR, "1_Data_Collection_and_Preprocessing", "valid")

# Model & training constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.35