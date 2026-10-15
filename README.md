# SignVision ISL: Dynamic Multi-Sign Recording & Recognition

A lightweight, CPU-only system for Indian Sign Language (ISL) recognition on Fedora Linux.
Supports dynamic data collection, training, and testing for **10+ arbitrary sign classes** without modifying source code for new signs.

---

## 1. Environment Setup

Activate the Python virtual environment:

```bash
cd /home/pruthvi/Documents/test
source .venv/bin/activate
```

Verify that all dependencies are installed:

```bash
python3 -c "import cv2, mediapipe, sklearn, joblib, numpy; print('All dependencies available!')"
```

The MediaPipe Tasks asset (`models/hand_landmarker.task`) is downloaded and reused across all scripts.

---

## 2. Interactive Data Collection (`record_videos.py`)

Run the multi-sign video recorder:

```bash
python3 record_videos.py
```

### On-Screen Interface & Controls:
- **Active Sign Display**: Shows the currently selected sign and the number of clips recorded for it.
- **`+ NEW SIGN` Button**:
  - Click the **`[+ NEW SIGN]`** button or the text field with your mouse (or press **`N`** on your keyboard).
  - Type the desired sign name (e.g. `THANK_YOU`, `HELP`, `WATER`, `HELLO`, `SORRY`).
  - Press **`[ENTER]`** to confirm.
  - The script validates the name, automatically creates `data/raw/<SIGN_NAME>/`, and sets it as active.
- **`● RECORD EXAMPLE (3s)` Button**:
  - Click the **`[● RECORD EXAMPLE]`** button with your mouse or press **`[SPACE]`** / **`[R]`**.
  - Immediately begins recording one 3-second clip with a red `REC [●]` indicator and on-screen progress bar.
  - Automatically saves the clip into `data/raw/<ACTIVE_SIGN>/<SIGN>_<timestamp>_<uuid>.mp4`.
  - Appends metadata to `data/metadata.csv` and increments the on-screen clip counter immediately.
- **Keyboard Shortcuts**:
  - `[SPACE]` / `[R]`: Trigger 3s recording for active sign
  - `[N]`: Activate text box to enter a new sign name
  - `[ENTER]`: Confirm typed sign name
  - `[ESC]`: Cancel text editing, or Quit if not typing
  - `[Q]`: Quit application cleanly

---

## 3. Dynamic Multi-Class Training Pipeline (`train_model.py`)

Once you have recorded clips for your desired signs (e.g., 5 to 10+ signs), train the unified multi-class model:

```bash
python3 train_model.py
```

### Dynamic Features:
- **Automatic Class Discovery**: Scans `data/raw/` dynamically. Any directory with video files becomes an active class (supports 2, 5, 10, 20+ classes without code edits).
- **Safe Backups**: Automatically backs up previous model artifacts (`sign_classifier.joblib.bak`, `model_metadata.json.bak`) before saving new ones.
- **Feature Extraction**: Extracts 16 uniformly sampled frames $\times$ 154 features ($77 \times 2$ hands) = 2,464 dimensions per clip via `src.landmark_features`.
- **Adaptive Cross-Validation**: Uses dynamic `StratifiedKFold` matching class counts to prevent fold errors.
- **Evaluation Metrics**: Reports overall accuracy, macro-F1, per-class precision/recall, and a full $K \times K$ confusion matrix.
- **Atomic Saving**: Uses temporary files and `os.replace` to guarantee artifact integrity.

### Optional Training Flags:
```bash
python3 train_model.py --cv-folds 5 --force-reextract
```

---

## 4. Multi-Class Webcam Testing (`test_webcam.py`)

Test live webcam gesture recognition across all trained signs:

```bash
python3 test_webcam.py
```

### Workflow:
1. **Live Preview**: See camera feed with real-time hand skeleton overlay and hand detection counts.
2. **Record & Predict**: Press **`[SPACE]`**. Captures a 3-second test clip, extracts 16 uniform frames, and predicts the sign.
3. **Multi-Class Output**: Displays the top predicted sign and ranked probabilities for the top candidate classes.
4. **Repeat**: Press **`[SPACE]`** to test another sign without restarting.

### Keyboard Controls:
| Key | Action |
| :--- | :--- |
| `[SPACE]` | Record 3.0s test sign and predict |
| `H` / `h` | Toggle hand skeleton overlay on/off |
| `S` / `s` | Save diagnostic screenshot to `screenshots/` |
| `C` / `c` | Clear previous prediction card |
| `Q` / `q` / `Esc` | Quit cleanly |

---

## 5. Saved Artifacts

- `artifacts/sign_classifier.joblib`: Multi-class classification pipeline (`StandardScaler` + `SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True)`).
- `artifacts/model_metadata.json`: Trained class names, label index mappings, feature dimensions, and package versions.
- `artifacts/training_report.json`: Multi-class cross-validation metrics, per-class F1, and confusion matrix.
- `models/hand_landmarker.task`: MediaPipe HandLandmarker Tasks asset.

---

## 6. Known Limitations

- **Closed-Set Vocabulary**: The classifier assigns probabilities across known trained signs. Unfamiliar or random gestures may still receive partial probability mass among the known classes; thresholding and hand coverage filters reduce false positives but cannot serve as a universal open-set detector.
- **Signer Diversity**: Cross-validation on recordings from a single user measures gesture consistency rather than generalization to new signers with different hand sizes or signing styles.
