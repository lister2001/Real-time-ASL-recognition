# Hand Sign (ASL Alphabet) Recognition

A real-time hand sign recognition system that uses **MediaPipe Hands** to extract 21 hand landmarks from a webcam feed and a lightweight **TensorFlow/Keras neural network** to classify them into the 26 letters of the ASL alphabet (A–Z). Includes both a live camera application and a full training pipeline for the classifier.

## How It Works

1. **Hand detection & landmark extraction** — MediaPipe Hands detects up to 2 hands per frame and returns 21 (x, y) keypoints per hand.
2. **Preprocessing** — Landmarks are converted to coordinates relative to the wrist (landmark 0), flattened into a 42-length vector (21 points × 2), and normalized by the maximum absolute value so the model is invariant to hand position/scale in the frame.
3. **Classification** — The 42-value vector is fed into a small dense neural network (or its quantized `.tflite` equivalent for fast inference) which outputs a softmax over 26 classes (A–Z).
4. **Display** — The predicted letter, hand bounding box, and skeleton overlay are drawn on the live video feed along with FPS.

## Project Structure

```
.
├── app.py                          # Main application: live inference / data collection
├── keypoint_classification.ipynb   # Training notebook for the classifier
├── keypoint_classifier.py          # TFLite wrapper class used by app.py at inference time
├── keypoint_classifier.keras       # Trained Keras model (full precision)
├── keypoint_classifier.tflite      # Quantized model used for real-time inference
├── keypoint_classifier_label.csv   # Class labels (A–Z)
├── cvfpscalc.py                    # Utility to compute rolling FPS
└── requirements.txt                # Python dependencies
```

> **Note on paths:** `app.py` imports from `utils.cvfpscalc` and `model.keypoint_classifier.keypoint_classifier`, and expects the label CSV, `.tflite` model, and dataset CSV under `model/keypoint_classifier/`. When setting up the repo, arrange the files into that structure (see [Setup](#setup) below) or update the import/path strings to match wherever you place them.

## Model Architecture

A compact fully-connected network trained on the 42-dimensional (21 landmarks × x,y) normalized keypoint vectors:

```
Input(42)
→ BatchNormalization
→ Dense(128, activation='mish', L2=0.01) → Dropout(0.5)
→ Dense(64,  activation='mish', L2=0.01) → Dropout(0.5)
→ Dense(32,  activation='mish', L2=0.01)
→ Dense(26,  activation='softmax')
```

- **Params:** ~16.9K total (~16.8K trainable) — small enough for real-time CPU inference
- **Optimizer / Loss:** Adam / sparse categorical crossentropy
- **Training:** up to 1000 epochs, batch size 128, with `ModelCheckpoint` and `EarlyStopping` (patience 20)
- **Train/test split:** 75/25, `random_state=42`
- **Deployment format:** converted to a quantized TensorFlow Lite model (`keypoint_classifier.tflite`) via `TFLiteConverter` with default optimizations, for lightweight real-time inference in `app.py`

**Reported test performance:** ~86.1% accuracy (loss ≈ 0.74) on the held-out split.

## Setup

1. **Clone/arrange the project** so the expected directory layout exists:
   ```
   model/
     keypoint_classifier/
       keypoint_classifier.py
       keypoint_classifier.keras
       keypoint_classifier.tflite
       keypoint_classifier_label.csv
       keypoint.csv           # training data, generated/collected via app.py
   utils/
     cvfpscalc.py
   app.py
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Key libraries: `opencv-python`, `mediapipe`, `tensorflow==2.16.1`, `numpy`, `pandas`, `scikit-learn`, `seaborn`, `matplotlib`, `Pillow`.

## Usage

### Run live recognition
```bash
python app.py
```
Optional flags:
- `--device` — camera index (default `0`)
- `--width`, `--height` — capture resolution (default `960x540`)
- `--min_detection_confidence`, `--min_tracking_confidence` — MediaPipe thresholds
- `--use_static_image_mode` — treat each frame as a standalone image (useful for stills)

Press **ESC** to quit the live window.

### Collect training data
`app.py` doubles as a data-collection tool with three modes, switched via keypress while the window is focused:
- **n** — Inference mode (default): shows live predictions.
- **k** — Key-point logging mode: press a letter key **A–Z** (mapped to labels 0–25) while showing a hand sign to append normalized landmark rows to `model/keypoint_classifier/keypoint.csv`.
- **d** — Bulk dataset mode: reads images from a folder structure at `model/dataset/dataset 1/<class_name>/*.jpg`, runs them through MediaPipe, and logs the resulting landmarks to the same CSV — useful for building a dataset from a folder of labeled images rather than live capture.

### Retrain the classifier
Open and run `keypoint_classification.ipynb` end-to-end:
1. Reads `model/keypoint_classifier/keypoint.csv` (columns: label, then 42 landmark values).
2. Splits into train/test (75/25).
3. Builds, trains, and evaluates the dense network described above.
4. Saves the trained model to `keypoint_classifier.keras`.
5. Converts and quantizes it to `keypoint_classifier.tflite` for use in `app.py`.
6. Includes a confusion matrix / classification report cell for per-class error analysis.

## Tech Stack

- **Computer vision:** OpenCV, MediaPipe (Hands solution)
- **Modeling:** TensorFlow / Keras (training), TensorFlow Lite (inference)
- **Data/eval:** NumPy, pandas, scikit-learn, seaborn, matplotlib

## Notes / Possible Next Steps

- Letters that involve motion in real ASL (notably **J** and **Z**) are inherently hard for a single-frame, static-keypoint classifier — worth checking per-class accuracy in the confusion matrix for these.
- ~86% test accuracy leaves room for improvement: more/varied training samples per class, data augmentation (small rotations/translations before landmark extraction), or hyperparameter tuning could help.
- `finger_gesture_id` in `app.py` is currently hardcoded to `0` — a second, unused classification hook that could be wired up for dynamic gesture recognition later.