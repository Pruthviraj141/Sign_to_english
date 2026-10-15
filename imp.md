# SignVision — Upgrade the Existing Project to Dynamic Multi-Sign ISL Recognition

## Objective

I have a working Fedora Linux, CPU-only prototype that recognizes two custom Indian Sign Language (ISL) signs:

- GOOD_MORNING
- GOOD_NIGHT

I have already recorded approximately 23 clips per sign, trained an SVM classifier using MediaPipe hand landmarks, and tested the existing model successfully with a live webcam.

Now I want to upgrade the entire existing project so I can collect, train, and recognize approximately 10 different signs, with the ability to add more signs later without modifying the source code for each new class.

I am developing this alone and need a reliable, maintainable implementation without unnecessary complexity.

## CRITICAL INSTRUCTIONS

1. Inspect the entire existing project before changing anything.
2. Preserve the working architecture, existing recordings, and useful preprocessing code.
3. Do not delete or overwrite the original GOOD_MORNING and GOOD_NIGHT recordings.
4. Do not redesign the application as a website or replace the existing OpenCV camera window.
5. Do not use pretrained sign-classification checkpoints or train large deep-learning models.
6. Reuse MediaPipe Hand Landmarker and the existing lightweight scikit-learn classifier approach.
7. Do not run a full model retraining or launch the live webcam test automatically during this modification task. I will collect more signs before training again.
8. Implement the necessary code now, perform safe static and import checks, and give me the exact commands to use later.

## Existing project components

Inspect the actual filenames and paths before editing. The project currently contains components similar to:

- `record_videos.py`
- `train_model.py`
- `test_webcam.py`
- `src/landmark_features.py`
- `data/raw/GOOD_MORNING/`
- `data/raw/GOOD_NIGHT/`
- `data/metadata.csv`
- `models/hand_landmarker.task`
- `artifacts/sign_classifier.joblib`
- `artifacts/model_metadata.json`
- `artifacts/training_report.json`

Do not assume every file or directory is exactly as listed. Discover the actual project structure and adapt to it.

---

# PART 1 — Improve the Recording Application

Modify the existing recording application rather than unnecessarily creating a second independent implementation.

## Interface

Keep the existing OpenCV camera preview as the main interface.

Draw a clean, readable control panel over the camera preview with:

- An editable sign-name field.
- A clickable `NEW SIGN` button.
- A clickable `RECORD EXAMPLE` button.
- The active sign name.
- The number of saved clips for the active sign.
- The total number of clips across all signs.
- A recording countdown and clear recording indicator.
- Short keyboard instructions.
- Useful error and success messages.

Implement the buttons through OpenCV mouse callbacks and the text field through keyboard input handled by the OpenCV event loop. Do not introduce a web framework or a complicated GUI framework.

The controls must actually work. Clicking a drawn button must invoke the corresponding action.

## Expected workflow

### Starting the application

Run the recording application and open the existing camera preview.

The user should be able to enter or select a sign name without editing Python source code.

For example, type:

`GOOD_MORNING`

Confirm the sign name, and make it the active class.

### Recording an example

When I click `RECORD EXAMPLE`:

1. Validate that an active sign name has been selected.
2. Show a brief countdown.
3. Record exactly one clip of approximately 3 seconds.
4. Save the original webcam video, without synthesizing or duplicating examples.
5. Display a recording indicator while capturing.
6. Save the clip under the corresponding class directory.
7. Update the clip count immediately after a successful save.
8. Return to the normal live preview automatically.

I want one clip per button click. Never begin recording successive clips automatically.

I should be able to record another example of the same sign simply by clicking the button again.

The current sign must remain selected until I explicitly click `NEW SIGN`.

### Adding a new sign

When I click `NEW SIGN`:

1. Allow me to enter a new sign name in the on-screen text field.
2. Normalize and validate the class name.
3. Confirm the new active sign.
4. Create its data directory automatically when necessary.
5. Preserve every other class and its existing recordings.
6. Allow immediate recording of its examples.

I must not have to edit `record_videos.py`, label mappings, or lists of supported classes whenever I add a new sign.

A practical naming convention is uppercase labels with underscores, such as `GOOD_MORNING`, `GOOD_NIGHT`, or `THANK_YOU`.

Reject empty names, path traversal, and invalid directory names. Handle duplicate names safely: selecting an existing class should allow additional recordings without overwriting its existing files.

## Recording reliability

- Save clips using unique filenames.
- Preserve the existing recording format when compatible.
- Keep the frame rate and preprocessing conventions compatible with the training pipeline.
- Preserve the existing metadata CSV structure where possible.
- Append new metadata rows safely without deleting historical records.
- Maintain labels, filenames, timestamps, and session information.
- Use a consistent session ID for each recording session.
- Handle camera initialization failures, failed writes, and interrupted recordings gracefully.
- Release video writers correctly.
- Prevent buttons from being activated accidentally during active recording.
- Do not lose existing recordings if the application is restarted.

Do not silently rewrite or remove old metadata rows. If existing metadata refers to a missing video file, report the discrepancy without deleting the row.

---

# PART 2 — Make the Training Pipeline Fully Multi-Class

Upgrade `train_model.py` to support any number of sign classes discovered from the actual dataset.

The initial intended vocabulary is approximately 10 signs, but do not hardcode a limit of 10.

## Dynamic dataset discovery

Discover class directories inside the existing raw dataset directory.

For example:

`data/raw/GOOD_MORNING/`

`data/raw/GOOD_NIGHT/`

`data/raw/THANK_YOU/`

`data/raw/HELP/`

Additional class directories must be discovered automatically.

Requirements:

- Do not hardcode two class names.
- Do not assume there are exactly two classes.
- Do not assume class IDs remain unchanged when a new class is added.
- Validate class names, video files, class counts, and metadata consistency.
- Report corrupted videos, missing files, and extraction failures.
- Require at least two usable classes for training.
- Warn when any class has too few examples to support meaningful evaluation.
- Never create artificial examples just to satisfy a class-count requirement.

## Preserve the existing feature extraction

Reuse `src/landmark_features.py` and the existing MediaPipe model.

The current trained model expects:

- 16 sampled frames per clip.
- 154 features per frame.
- 2,464 features for each flattened sequence.

Verify these dimensions against the real metadata and implementation.

Keep the existing compatible feature format unless a demonstrated bug requires a correction. If a feature-format change is necessary, explain why and ensure both training and live inference use the same corrected implementation.

Preserve the handling of:

- Both hands and their handedness.
- The 21 landmarks of each detected hand.
- Missing-hand frames.
- Coordinate normalization.
- Frame ordering.
- Temporal resampling.
- Image color conversion.
- Mirroring and orientation conventions.

Treat every video as one training sample. Never randomly divide adjacent frames from the same video into separate train and test examples.

## Classifier

Keep the system lightweight for Fedora CPU-only execution.

Start with the existing StandardScaler and RBF SVM pipeline. Use the actual number of classes and balanced class weighting where appropriate.

Use a consistent, deterministic label mapping derived from the dataset and save it with the model. Respect the classifier's actual `classes_` ordering when associating predicted probabilities with labels.

The system must support 3, 5, 10, 20, or more classes without source-code changes.

Do not introduce PyTorch, TensorFlow, Transformers, or unnecessary dependencies.

## Evaluation

For the multi-class training pipeline, report:

- Number of usable recordings per class.
- Extraction failures and hand-detection coverage.
- Cross-validation accuracy.
- Macro-F1.
- Per-class precision, recall, and F1.
- A multi-class confusion matrix.
- Fold-to-fold variability.
- Session/signer leakage limitations.

Use video-level stratified evaluation where suitable. Use grouped evaluation when valid signer or recording-session metadata permits it. If grouped evaluation is infeasible, disclose the reason rather than pretending the model has been tested on unseen signers.

Handle classes with insufficient examples gracefully. Never report fabricated accuracy or treat training accuracy as evidence of real-world webcam performance.

Before replacing an existing trained artifact, preserve a backup of the working model and its metadata. Do not overwrite the current working two-class artifact merely because the code has been upgraded.

## Artifact compatibility

Save the updated multi-class model and all required metadata so the live webcam program can load it directly.

The metadata must include:

- Class names and exact label mapping.
- Feature dimensions and frame sequence length.
- Landmark preprocessing configuration.
- MediaPipe model asset location.
- Model version and relevant package versions.
- Evaluation report and dataset counts.

Use safe, atomic artifact writes.

The live application must never silently combine a model with incompatible metadata or preprocessing.

---

# PART 3 — Upgrade the Live Webcam Recognition Script

Modify `test_webcam.py` to work with the dynamic multi-class model.

Do not run it automatically during this task.

## Dynamic prediction

Remove assumptions that the only possible outputs are GOOD_MORNING and GOOD_NIGHT.

The script must obtain its supported sign labels from the saved model and metadata.

Requirements:

1. Load the trained model and matching metadata.
2. Verify all model and MediaPipe assets.
3. Use exactly the same landmark preprocessing as training.
4. Maintain the required rolling temporal window.
5. Classify the complete sequence.
6. Map predictions and probabilities to the correct class names.
7. Smooth predictions temporally to reduce flickering.
8. Display the current prediction, confidence, and stability information.
9. Show `UNCERTAIN` when the configured decision criteria are not satisfied.
10. Keep the webcam responsive on CPU-only hardware.

For the visual display, show the predicted word prominently, with the top few candidate classes and their scores where useful.

Preserve the existing controls for quitting, resetting the prediction buffer, and toggling landmark visualization if already implemented.

## Hand and finger visualization

Keep drawing the correct hand-landmark connections for the thumb and all four fingers.

Show the detected hand count, landmark availability, latency, and FPS.

Handle missing hands without crashing or reusing stale landmarks as if they were new observations.

## Unknown signs and uncertainty

Understand that a classifier trained only on known classes can still confidently assign an unfamiliar gesture to one of them.

A confidence threshold is not a guarantee of unknown-sign rejection.

Implement conservative uncertainty behavior and document that genuine unknown-sign rejection requires suitable negative examples and separate evaluation.

Do not claim that the system supports unrestricted ISL translation. It classifies isolated signs from its trained vocabulary.

---

# PART 4 — Maintainability and Project Organization

Inspect the existing project and avoid unnecessary architectural changes.

Keep one shared preprocessing implementation used by training and live inference.

Separate responsibilities clearly:

- Recording: collect labeled real video examples.
- Feature extraction: convert videos or frames into the agreed landmark representation.
- Training: discover classes, train, evaluate, and save the model.
- Webcam inference: load the saved model and recognize known classes.

Add reusable utility functions where they reduce duplicated logic.

Avoid circular imports and hardcoded absolute home-directory paths. Use project-relative paths based on the script location or an explicitly configured project root.

Update the README with:

- Environment activation.
- Recording instructions.
- How to add any new class.
- How to collect multiple examples.
- How to train the multi-class model.
- How to run live webcam detection.
- Artifact locations.
- Keyboard and mouse controls.
- Known limitations.

Do not require the user to update a list of labels manually for each new sign.

## Execution policy for this task

Implement the code now, but do not automatically record videos, retrain the model, or launch webcam inference.

Perform safe syntax and import checks where possible. Inspect the final source for hardcoded two-class assumptions, inconsistent preprocessing, incorrect class-probability mappings, and destructive file operations.

If a missing dependency is discovered, explain it before installing any unnecessary package.

Do not claim real model accuracy or live webcam functionality was verified during this modification task.

## Final response

Tell me:

1. Which files were modified or created.
2. How to launch the recording application.
3. How to enter a sign name and record one 3-second example.
4. How to switch to another sign without losing previous data.
5. The exact command to train all classes later.
6. The exact command to launch multi-class webcam detection after training.
7. Any compatibility issues or remaining manual steps.

My immediate goal is to collect data for approximately 10 signs through a simple OpenCV interface, then train one model that recognizes all collected classes. Prioritize correctness, ease of adding new signs, CPU performance, and preserving the existing working system.