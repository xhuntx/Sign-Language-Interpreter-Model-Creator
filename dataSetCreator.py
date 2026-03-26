import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split

DATASET_PATH = "model/Training"
OUTPUT_DIR = "processed_dataset"
MODEL_PATH = "hand_landmarker.task"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_labels_and_paths(root_dir):
    raw_labels = []
    image_paths = []

    for label_name in sorted(os.listdir(root_dir)):
        label_path = os.path.join(root_dir, label_name)
        if not os.path.isdir(label_path):
            continue
        if not label_name:
            continue

        for fname in os.listdir(label_path):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            image_paths.append(os.path.join(label_path, fname))
            raw_labels.append(label_name)

    return np.array(image_paths), np.array(raw_labels, dtype=object)


def create_hand_landmarker(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"HandLandmarker model not found at: {model_path}.")

    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
    )

    return HandLandmarker.create_from_options(options)


def extract_hand_landmarks_with_tasks(image_bgr, landmarker):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = landmarker.detect(mp_image)

    if not detection_result.hand_landmarks:
        return None

    hand_landmarks = detection_result.hand_landmarks[0]
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])

    return np.array(coords, dtype=np.float32)


def main():
    print(f"Loading images from: {DATASET_PATH}")
    image_paths, raw_labels = get_labels_and_paths(DATASET_PATH)
    print(f"Found {len(image_paths)} images.")

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found under {DATASET_PATH}.")

    print(f"Loading HandLandmarker model from: {MODEL_PATH}")
    landmarker = create_hand_landmarker(MODEL_PATH)

    all_features = []
    all_labels = []
    no_hand_count = 0

    for img_path, label_name in zip(image_paths, raw_labels):
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: could not read image {img_path}, skipping.")
            continue

        features = extract_hand_landmarks_with_tasks(image, landmarker)
        if features is None:
            no_hand_count += 1
            continue

        all_features.append(features)
        all_labels.append(label_name)

    landmarker.close()

    if not all_features:
        raise RuntimeError("No hands detected in any image using HandLandmarker.")

    X = np.stack(all_features)
    raw_labels = np.array(all_labels, dtype=object)

    unique_labels = np.unique(raw_labels)
    unique_labels_sorted = np.sort(unique_labels)

    label_to_id = {label: idx for idx, label in enumerate(unique_labels_sorted)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    y_encoded = np.array([label_to_id[label] for label in raw_labels], dtype=np.int32)

    num_classes = len(unique_labels_sorted)
    n_samples = len(y_encoded)

    print("Label mapping (string -> id):", label_to_id)
    print(f"Total samples: {n_samples}, classes: {num_classes}")

    unique_ids, counts = np.unique(y_encoded, return_counts=True)
    print("Counts per class (id):", dict(zip(unique_ids, counts)))

    # save mapping as an array so Pylance is happy
    mapping_array = np.array(
        [[idx, label] for idx, label in id_to_label.items()],
        dtype=object,
    )
    np.save(os.path.join(OUTPUT_DIR, "id_to_label.npy"), mapping_array)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.5,
        random_state=42,
    )

    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    print(f"Saved dataset to {OUTPUT_DIR}")
    print("Train:", X_train.shape, "Test:", X_test.shape)
    print(f"Skipped {no_hand_count} images with no detected hands.")


if __name__ == "__main__":
    main()
