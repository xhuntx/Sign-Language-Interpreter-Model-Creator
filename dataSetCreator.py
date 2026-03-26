import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split

DATASET_PATH = "model/Training"
OUTPUT_DIR = "processed_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_labels_and_paths(root_dir):
    labels = []
    image_paths = []
    for label_name in sorted(os.listdir(root_dir)):
        label_path = os.path.join(root_dir, label_name)
        if not os.path.isdir(label_path):
            continue
        # Only accept numeric labels (1, 2, ..., 10)
        try:
            label_int = int(label_name)
        except ValueError:
            continue

        for fname in os.listdir(label_path):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            image_paths.append(os.path.join(label_path, fname))
            labels.append(label_int)

    return np.array(image_paths), np.array(labels)


def extract_hand_landmarks(image_bgr, hands):
    # Convert BGR -> RGB for Mediapipe
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    # Use the first detected hand
    hand_landmarks = results.multi_hand_landmarks[0]
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])

    # coords length = 21 landmarks * 3 coords = 63
    return np.array(coords, dtype=np.float32)


def main():
    print(f"Loading images from: {DATASET_PATH}")
    image_paths, labels = get_labels_and_paths(DATASET_PATH)
    print(f"Found {len(image_paths)} images.")

    if len(image_paths) == 0:
        raise RuntimeError(
            f"No images found under {DATASET_PATH}. "
            f"Expected folders like model/Training/1-10/1, 2, ... with .png/.jpg files."
        )

    mp_hands = mp.solutions.hands

    all_features = []
    all_labels = []
    no_hand_count = 0

    # static_image_mode=True -> treat each image independently (no webcam)
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands:
        for img_path, label in zip(image_paths, labels):
            image = cv2.imread(img_path)
            if image is None:
                continue

            features = extract_hand_landmarks(image, hands)
            if features is None:
                # No hand detected; skip this sample
                no_hand_count += 1
                continue

            all_features.append(features)
            all_labels.append(label)

    if not all_features:
        raise RuntimeError(
            "No hands detected in any image. "
            "Either your images don't clearly show hands, or Mediapipe Hands isn't working correctly."
        )

    X = np.stack(all_features)
    y = np.array(all_labels, dtype=np.int32)

    print(f"Kept {X.shape[0]} samples after hand detection (skipped {no_hand_count}).")
    print(f"Feature vector shape per sample: {X.shape[1]}")

    # Normalize labels from 1–10 to 0–9 for one-hot training later
    y_zero_based = y - 1

    num_classes = len(np.unique(y_zero_based))
    n_samples = len(y_zero_based)
    print(f"Total samples: {n_samples}, classes: {num_classes}")

    # DEBUG: show how many samples you have per class
    unique, counts = np.unique(y_zero_based, return_counts=True)
    print("Counts per class (0–9):", dict(zip(unique, counts)))

    # Random train/test split WITHOUT stratify, because with 2 images per class
    # a stratified 80/20 split is mathematically impossible (test set too small).
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_zero_based,
        test_size=0.2,
        random_state=42
        # no stratify here
    )

    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    print(f"Saved dataset to {OUTPUT_DIR}")
    print("Train:", X_train.shape, "Test:", X_test.shape)


if __name__ == "__main__":
    main()
