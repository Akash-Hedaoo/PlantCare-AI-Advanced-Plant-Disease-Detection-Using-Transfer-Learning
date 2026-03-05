# 2_Model_Building/utils.py
import os

def get_num_classes(train_dir):
    """Return number of class folders inside train_dir and a dict of counts."""
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    class_counts = {}
    total = 0
    for c in classes:
        count = len([f for f in os.listdir(os.path.join(train_dir, c)) if os.path.isfile(os.path.join(train_dir, c, f))])
        class_counts[c] = count
        total += count
    return len(classes), class_counts, total

if __name__ == "__main__":
    # quick test when run directly
    import config
    n_classes, counts, total = get_num_classes(config.TRAIN_DIR)
    print("Num classes:", n_classes)
    print("Total images in train:", total)
    print("Sample class counts (first 10):")
    for k,v in list(counts.items())[:10]:
        print(k, v)