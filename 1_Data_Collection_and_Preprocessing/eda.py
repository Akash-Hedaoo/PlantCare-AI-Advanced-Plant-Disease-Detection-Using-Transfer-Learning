# ===============================
# Activity 1.2: Data Exploration
# ===============================

import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# ===============================
# 1. Configure Dataset Paths
# ===============================

train_dir = "train"
valid_dir = "valid"

print("Train path exists:", os.path.exists(train_dir))
print("Validation path exists:", os.path.exists(valid_dir))


# ===============================
# 2. Visualize Sample Images
# ===============================

def show_samples(directory, num_classes=3, images_per_class=3):
    classes = os.listdir(directory)
    selected_classes = random.sample(classes, num_classes)

    plt.figure(figsize=(10, 8))
    i = 1

    for cls in selected_classes:
        class_path = os.path.join(directory, cls)

        # select random images
        available_images = os.listdir(class_path)
        num_to_sample = min(len(available_images), images_per_class)
        if num_to_sample == 0:
            continue
        images = random.sample(available_images, num_to_sample)

        for img in images:
            img_path = os.path.join(class_path, img)
            image = Image.open(img_path)

            plt.subplot(num_classes, images_per_class, i)
            plt.imshow(image)
            plt.title(cls.replace("_", " "))
            plt.axis("off")
            i += 1

    plt.suptitle("Sample Images from Dataset", fontsize=16)
    plt.tight_layout()
    plt.show()


print("\nShowing sample images...")
show_samples(train_dir)


# ===============================
# 3. Dataset Statistics
# ===============================

def count_images(directory):
    total_images = 0
    class_counts = {}

    for cls in os.listdir(directory):
        class_path = os.path.join(directory, cls)

        if os.path.isdir(class_path):
            num_images = len(os.listdir(class_path))
            class_counts[cls] = num_images
            total_images += num_images

    return total_images, class_counts


# get statistics
train_total, train_counts = count_images(train_dir)
valid_total, valid_counts = count_images(valid_dir)

print("\n===== DATASET STATISTICS =====")
print("Training images:", train_total)
print("Validation images:", valid_total)
print("Number of classes:", len(train_counts))

print("\nClass distribution (first 10):")
for cls, count in list(train_counts.items())[:10]:
    print(cls, ":", count)