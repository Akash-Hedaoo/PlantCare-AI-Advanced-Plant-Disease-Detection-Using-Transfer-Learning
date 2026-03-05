from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# Only normalization for validation
valid_gen = ImageDataGenerator(rescale=1./255)

# Load training data
train_data = train_gen.flow_from_directory(
    "train",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

# Load validation data
valid_data = valid_gen.flow_from_directory(
    "valid",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

print("Classes:", train_data.class_indices)