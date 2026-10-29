# Sign Language Interpreter: Model Creator

This is the training pipeline for the [Sign Language Interpreter](https://github.com/xhuntx/Sign-language-interpreter). Use it to retrain the hand-sign classifier, add new signs, or improve accuracy, then drop the new model into the main app.

The training images come with the repo, so you can install the dependencies and train a model right away.

[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)  ![Python](https://img.shields.io/badge/python-3.11-blue)

---

## Table of Contents

- [How It Works](#how-it-works)
- [Included Training Data](#included-training-data)
- [Getting Started](#getting-started)
- [Training a Model](#training-a-model)
- [Adding a New Sign](#adding-a-new-sign)
- [Using Your Model in the Interpreter](#using-your-model-in-the-interpreter)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## How It Works

```
Images (one folder per sign) ──► MediaPipe HandLandmarker ──► 126-value feature vectors ──► Train/test split ──► Keras classifier ──► sign_language_classifier.h5
```

1. **`dataSetCreator.py`** reads every image in `model/Training/<label>/`. It runs MediaPipe's `HandLandmarker` on each one to find up to 2 hands, with 21 landmarks per hand.
2. The `(x, y, z)` landmarks are packed into a **126-value vector**: left hand = values `0–62`, right hand = values `63–125`. A missing hand is filled with zeros.
3. The data is split **80/20** into train and test sets and saved as `.npy` files in `processed_dataset/`.
4. **`datasetToH5.py`** trains a small dense network (`128 → 64 → softmax`) for 30 epochs and saves it as `sign_language_classifier.h5`.

---

## Included Training Data

The repo includes a ready-to-use training set in `model/Training/`, with **one folder per sign**. The folder name is the label.

| Category | Folders |
| --- | --- |
| Letters | `a` – `z` (ASL fingerspelling) |
| Numbers | `0` – `10` |
| Other | `none` (no sign / neutral hand) |

That's **38 classes**, the same set the interpreter app recognizes.

---

## Getting Started

### Prerequisites

- Python 3.11

You don't need to supply any images. They're already in `model/Training/`.

### Installation

```bash
git clone https://github.com/xhuntx/Sign-Language-Interpreter-Model-Creator.git
cd Sign-Language-Interpreter-Model-Creator

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
| --- | --- | --- |
| `mediapipe` | 0.10.33 | Hand landmark detection |
| `opencv-python` | 4.13.0.92 | Image loading |
| `numpy` | 2.4.3 | Feature arrays |
| `scikit-learn` | 1.8.0 | Train/test split |
| `tensorflow` | 2.21.0 | Model training |

---

## Training a Model

Run both scripts from the repo root.

### 1. Extract landmarks

```bash
python dataSetCreator.py
```

This turns the images in `model/Training/` into feature vectors. It writes `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`, and `id_to_label.npy` to `processed_dataset/`.

It also prints:
- the label mapping (label → ID)
- the number of samples per class
- how many images were skipped because no hand was detected

### 2. Train the classifier

```bash
python datasetToH5.py
```

This trains the model on `processed_dataset/` and saves `sign_language_classifier.h5` in the repo root.

---

## Adding a New Sign

1. Create a new folder in `model/Training/` next to the existing ones, named after the sign (for example, `hello/`).
2. Add `.png`, `.jpg`, or `.jpeg` images of the sign. Try to match the image count of the existing folders, and vary the lighting, backgrounds, and hand positions.
3. Run `dataSetCreator.py` and `datasetToH5.py` again.
4. Update `labels.json` in the interpreter repo (see below).

> **Note:** Images where MediaPipe can't detect a hand are skipped. Every image needs a visible hand, including the `none` class (use a relaxed, neutral hand rather than an empty frame).

---

## Using Your Model in the Interpreter

1. Copy `sign_language_classifier.h5` into the [Sign Language Interpreter](https://github.com/xhuntx/Sign-language-interpreter) repo.
2. If you added or removed classes, update `labels.json` there. **The label order must match the mapping printed by `dataSetCreator.py`.** Labels are sorted as text, so `10` comes before `2`.
3. Run `python main.py`.

---

## Project Structure

```
Sign-Language-Interpreter-Model-Creator/
├── dataSetCreator.py            # Images → landmark features → train/test .npy files
├── datasetToH5.py               # Trains the classifier and saves it as .h5
├── hand_landmarker.task         # MediaPipe hand landmark model
├── sign_language_classifier.h5  # Most recent trained model
├── model/
│   └── Training/                # Included training images (0–10, a–z, none)
├── processed_dataset/           # Generated .npy feature files
├── requirements.txt
└── LICENSE
```

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `No images found under model/Training` | Run the script from the repo root, and make sure images are inside label subfolders, not directly in `Training/`. |
| `HandLandmarker model not found` | Run the script from the repo root so `hand_landmarker.task` can be found. |
| Many images skipped | MediaPipe couldn't find a hand. Use clearer, well-lit images with the whole hand in frame. |
| Wrong predictions in the app | Check that the order in `labels.json` matches the printed label mapping. |
| Install errors | Make sure you're using Python 3.11 inside a fresh virtual environment. |

---

## Contributing

Pull requests are welcome, especially new training images and new signs.

1. Fork the repo
2. Create a branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

Made by [xhuntx](https://github.com/xhuntx)
