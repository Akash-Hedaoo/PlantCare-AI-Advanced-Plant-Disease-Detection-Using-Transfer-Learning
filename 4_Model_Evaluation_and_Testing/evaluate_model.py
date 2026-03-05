# evaluate_model.py

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw

# allow importing files from 2_Model_Building
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "2_Model_Building"))

import config
from utils import get_num_classes
from build_model import build_model

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf


# ---------------- PATH SETTINGS ----------------

BASE_DIR = os.path.dirname(__file__)
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "..", "6_Models_and_Outputs")

BEST_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, "plant_disease_best.h5")
FALLBACK_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, "plant_disease_final.h5")

SAVEDMODEL_DIR = os.path.join(MODEL_OUTPUT_DIR, "plant_disease_saved_model")
PREDICTIONS_CSV = os.path.join(MODEL_OUTPUT_DIR, "predictions.csv")
CONF_MATRIX_PNG = os.path.join(MODEL_OUTPUT_DIR, "confusion_matrix.png")
WRONG_EXAMPLES_DIR = os.path.join(MODEL_OUTPUT_DIR, "wrong_examples")

os.makedirs(WRONG_EXAMPLES_DIR, exist_ok=True)


# ---------------- BUILD MODEL ----------------

print("Detecting number of classes...")

num_classes, _, _ = get_num_classes(config.TRAIN_DIR)

print("Number of classes:", num_classes)

print("Rebuilding model architecture...")

model = build_model(num_classes, input_shape=(*config.IMG_SIZE, 3))


# ---------------- LOAD TRAINED WEIGHTS ----------------

model_path = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else FALLBACK_MODEL_PATH

if not os.path.exists(model_path):
    raise FileNotFoundError("No trained model found. Please run training first.")

print("Loading weights from:", model_path)

model.load_weights(model_path)

model.trainable = False


# ---------------- LOAD VALIDATION DATA ----------------

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

val_gen = val_datagen.flow_from_directory(
    config.VALID_DIR,
    target_size=config.IMG_SIZE,
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

inv_map = {v: k for k, v in val_gen.class_indices.items()}

num_samples = val_gen.samples

print("Validation samples:", num_samples)


# ---------------- RUN PREDICTIONS ----------------

print("Running predictions...")

pred_probs = model.predict(val_gen, verbose=1)

pred_indices = np.argmax(pred_probs, axis=1)

true_indices = val_gen.classes

filenames = val_gen.filenames


# ---------------- CLASSIFICATION REPORT ----------------

y_true = [inv_map[i] for i in true_indices]
y_pred = [inv_map[i] for i in pred_indices]

report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, zero_division=0))

with open(os.path.join(MODEL_OUTPUT_DIR, "classification_report.json"), "w") as f:
    json.dump(report, f, indent=2)


# ---------------- CONFUSION MATRIX ----------------

cm = confusion_matrix(true_indices, pred_indices, normalize='true')

plt.figure(figsize=(12,10))

sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=[inv_map[i] for i in range(len(inv_map))],
    yticklabels=[inv_map[i] for i in range(len(inv_map))],
    annot=False
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Normalized Confusion Matrix")

plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(CONF_MATRIX_PNG, dpi=150)

print("Saved confusion matrix:", CONF_MATRIX_PNG)

plt.close()


# ---------------- SAVE PREDICTIONS CSV ----------------

rows = []

for fn, t_idx, p_idx, probs in zip(filenames, true_indices, pred_indices, pred_probs):
    rows.append({
        "filename": fn,
        "true_label": inv_map[int(t_idx)],
        "pred_label": inv_map[int(p_idx)],
        "confidence": float(probs[int(p_idx)])
    })

df = pd.DataFrame(rows)

df.to_csv(PREDICTIONS_CSV, index=False)

print("Saved predictions CSV:", PREDICTIONS_CSV)


# ---------------- SAVE WRONG EXAMPLES ----------------

MAX_PER_CLASS = 5

wrong_counts = {}
saved = 0

for r in rows:

    if r["true_label"] != r["pred_label"]:

        cls = r["true_label"]

        wrong_counts.setdefault(cls, 0)

        if wrong_counts[cls] < MAX_PER_CLASS:

            src = os.path.join(config.VALID_DIR, r["filename"])

            dst = os.path.join(
                WRONG_EXAMPLES_DIR,
                f"{saved}_{Path(r['filename']).stem}_true-{r['true_label']}_pred-{r['pred_label']}.jpg"
            )

            try:

                im = Image.open(src).convert("RGB")

                draw = ImageDraw.Draw(im)

                txt = f"T:{r['true_label']} P:{r['pred_label']} ({r['confidence']:.2f})"

                draw.text((5,5), txt, fill=(255,0,0))

                im.save(dst)

                wrong_counts[cls] += 1
                saved += 1

            except Exception as e:

                print("Failed saving example:", e)


print("Wrong prediction examples saved:", saved)


# ---------------- SAVE MODEL FOR DEPLOYMENT ----------------

print("Saving model as SavedModel format...")

model.save(SAVEDMODEL_DIR, include_optimizer=False)

print("SavedModel stored at:", SAVEDMODEL_DIR)


print("\nEvaluation complete!")
print("Outputs saved in:", MODEL_OUTPUT_DIR)