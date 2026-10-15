#!/usr/bin/env python3
"""
SignVision Multi-Class Classifier Training Pipeline (`train_model.py`)

Trains a lightweight Sign Language recognition classifier for any number of ISL signs
discovered dynamically from the dataset (e.g., 2, 5, 10, or more signs).

Pipeline:
1. Dynamically discovers all class directories under data/raw/ containing video clips.
2. Cross-references clips with data/metadata.csv and reports discrepancies.
3. Extracts fixed-size landmark sequences (16 frames x 154 features = 2464 dims).
4. Assesses per-class hand detection coverage and flags poorly detected clips.
5. Evaluates classification pipelines with Stratified K-Fold Cross-Validation.
6. Fits final multi-class pipeline (StandardScaler + regularized RBF-kernel SVM with balanced class weights).
7. Preserves backups of existing model artifacts before writing new ones.
8. Atomically saves reusable artifacts to artifacts/ directory.
9. Performs post-save reload and prediction verification.
"""

import os
import sys
import json
import csv
import shutil
import argparse
import tempfile
import time
import warnings
from datetime import datetime
from collections import Counter
from typing import Dict, List, Tuple, Any

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import joblib
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from src.landmark_features import (
    HandLandmarkerExtractor,
    get_feature_config,
    DEFAULT_NUM_FRAMES,
    FEATURE_DIM_PER_FRAME,
    TOTAL_CLIP_FEATURE_DIM,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SignVision ISL Dynamic Multi-Class Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/raw",
        help="Root path containing sign class subdirectories",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default="data/metadata.csv",
        help="Path to metadata CSV file",
    )
    parser.add_argument(
        "--task-model",
        type=str,
        default="models/hand_landmarker.task",
        help="Path to MediaPipe hand_landmarker.task",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Directory to store model artifacts and reports",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Target number of cross-validation folds",
    )
    parser.add_argument(
        "--cache-features",
        type=str,
        default="artifacts/features_cache.joblib",
        help="Path to cached extracted features",
    )
    parser.add_argument(
        "--force-reextract",
        action="store_true",
        help="Force re-extraction of features from raw video files",
    )
    return parser.parse_args()


def inspect_dataset(data_root: str, metadata_csv_path: str):
    """
    Dynamically discovers all class subdirectories in data_root containing video files,
    cross-references metadata.csv, and checks class balance.
    """
    print(f"\n[INFO] Discovering dataset dynamically from: {data_root}")
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Data root directory not found: {data_root}")

    # Read metadata.csv if available
    metadata_map = {}
    if os.path.isfile(metadata_csv_path):
        with open(metadata_csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata_map[row["filename"]] = row
        print(f"[INFO] Loaded {len(metadata_map)} records from metadata CSV.")
    else:
        print(f"[WARN] Metadata CSV not found at {metadata_csv_path}.")

    discovered_files: Dict[str, List[str]] = {}
    total_found = 0

    candidate_dirs = sorted([
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d)) and not d.startswith(".")
    ])

    for cls in candidate_dirs:
        cls_dir = os.path.join(data_root, cls)
        files = sorted([
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".mp4", ".avi", ".mov"))
        ])
        if len(files) > 0:
            discovered_files[cls] = [os.path.join(cls_dir, f) for f in files]
            total_found += len(files)
            print(f"  Class '{cls}': {len(files)} video clips found on disk.")

    discovered_classes = sorted(discovered_files.keys())
    print(f"[INFO] Total discovered classes: {len(discovered_classes)} ({total_found} total clips)")

    if len(discovered_classes) < 2:
        raise ValueError(
            f"At least 2 sign classes with valid video clips are required for training. "
            f"Found {len(discovered_classes)}: {discovered_classes}"
        )

    for cls in discovered_classes:
        cnt = len(discovered_files[cls])
        if cnt < 3:
            print(f"[WARN] Class '{cls}' has only {cnt} clip(s). Cross-validation evaluation may be limited.")

    # Cross-reference with metadata
    csv_missing_on_disk = []
    disk_missing_in_csv = []
    for cls, paths in discovered_files.items():
        for p in paths:
            fname = os.path.basename(p)
            if fname not in metadata_map:
                disk_missing_in_csv.append(fname)

    if metadata_map:
        for fname in metadata_map:
            found = False
            for cls, paths in discovered_files.items():
                if any(os.path.basename(p) == fname for p in paths):
                    found = True
                    break
            if not found:
                csv_missing_on_disk.append(fname)

    if csv_missing_on_disk:
        print(f"[WARN] {len(csv_missing_on_disk)} record(s) in CSV do not exist on disk: {csv_missing_on_disk}")
    if disk_missing_in_csv:
        print(f"[WARN] {len(disk_missing_in_csv)} disk video(s) are not registered in CSV: {disk_missing_in_csv}")

    # Session analysis
    session_counts = Counter()
    for cls, paths in discovered_files.items():
        for p in paths:
            fname = os.path.basename(p)
            sess = metadata_map.get(fname, {}).get("session_id", "unknown_session")
            session_counts[(cls, sess)] += 1

    print("\n[INFO] Session distribution across classes:")
    for (cls, sess), count in session_counts.items():
        print(f"  - Class '{cls}' | Session '{sess}': {count} clips")

    unique_sessions = set(sess for (_, sess) in session_counts.keys())
    sessions_with_all_classes = [
        s for s in unique_sessions
        if all((cls, s) in session_counts for cls in discovered_classes)
    ]
    print(f"[INFO] Sessions containing all classes: {len(sessions_with_all_classes)} of {len(unique_sessions)}")

    return discovered_files, metadata_map, sessions_with_all_classes


def extract_features_dataset(
    discovered_files: Dict[str, List[str]],
    metadata_map: Dict[str, Dict[str, Any]],
    task_model_path: str,
    cache_path: str,
    force_reextract: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], List[str]]:
    """
    Extracts features for all video clips using HandLandmarkerExtractor.
    Checks cache validity against current class list and clip counts.
    """
    class_names = sorted(discovered_files.keys())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    total_clips = sum(len(f) for f in discovered_files.values())

    if os.path.isfile(cache_path) and not force_reextract:
        try:
            cached = joblib.load(cache_path)
            cached_classes = cached.get("class_names", [])
            cached_samples = len(cached.get("y", []))
            if sorted(cached_classes) == class_names and cached_samples == total_clips:
                print(f"[INFO] Loaded valid cached features ({cached_samples} clips, {len(class_names)} classes) from: {cache_path}")
                return cached["X"], cached["y"], cached["clip_metadata"], cached["class_names"]
            else:
                print(f"[INFO] Cache invalidated (cached: {cached_classes}, current: {class_names}). Re-extracting.")
        except Exception as e:
            print(f"[WARN] Failed to load cache ({e}). Re-extracting from raw videos.")

    print(f"\n[INFO] Initializing MediaPipe extractor from: {task_model_path}")
    extractor = HandLandmarkerExtractor(task_model_path)

    X_list = []
    y_list = []
    clip_metadata = []
    processed_count = 0
    failed_clips = []

    print(f"[INFO] Extracting features from {total_clips} clips across {len(class_names)} classes (16 frames/clip)...")
    start_time = time.time()

    for cls in class_names:
        paths = discovered_files[cls]
        for path in paths:
            processed_count += 1
            fname = os.path.basename(path)
            sess_id = metadata_map.get(fname, {}).get("session_id", "unknown")
            try:
                clip_vec, meta = extractor.extract_video_features(
                    path, num_sampled_frames=DEFAULT_NUM_FRAMES
                )
                meta["label"] = cls
                meta["label_idx"] = class_to_idx[cls]
                meta["session_id"] = sess_id
                meta["filename"] = fname

                X_list.append(clip_vec)
                y_list.append(class_to_idx[cls])
                clip_metadata.append(meta)

                sys.stdout.write(
                    f"\r[{processed_count}/{total_clips}] Extracted '{fname}' | Hands coverage: {meta['coverage_ratio']*100:.1f}%"
                )
                sys.stdout.flush()
            except Exception as e:
                print(f"\n[ERROR] Failed to extract from {path}: {e}", file=sys.stderr)
                failed_clips.append({"path": path, "error": str(e)})

    extractor.close()
    elapsed = time.time() - start_time
    print(f"\n[INFO] Extraction finished in {elapsed:.1f}s. Success: {len(X_list)}, Failed: {len(failed_clips)}")

    if failed_clips:
        print("[WARN] Failed clips summary:")
        for fc in failed_clips:
            print(f"  - {fc['path']}: {fc['error']}")

    if len(X_list) == 0:
        raise RuntimeError("No features could be extracted. Aborting.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    joblib.dump(
        {"X": X, "y": y, "clip_metadata": clip_metadata, "class_names": class_names},
        cache_path,
    )
    print(f"[INFO] Saved feature cache to: {cache_path}")

    return X, y, clip_metadata, class_names


def analyze_hand_coverage(clip_metadata: List[Dict[str, Any]], class_names: List[str]):
    """Analyzes and reports per-class hand detection coverage."""
    print("\n" + "=" * 65)
    print("HAND DETECTION COVERAGE ANALYSIS")
    print("=" * 65)

    low_coverage_clips = []
    coverage_by_class = {c: [] for c in class_names}

    for meta in clip_metadata:
        cls = meta["label"]
        ratio = meta["coverage_ratio"]
        coverage_by_class[cls].append(ratio)
        if ratio < 0.50:
            low_coverage_clips.append(meta)

    for cls in class_names:
        ratios = coverage_by_class[cls]
        if not ratios:
            continue
        mean_cov = np.mean(ratios) * 100
        min_cov = np.min(ratios) * 100
        max_cov = np.max(ratios) * 100
        print(f"  Class '{cls:16s}': {len(ratios):2d} clips | Mean: {mean_cov:5.1f}% | Min: {min_cov:5.1f}% | Max: {max_cov:5.1f}%")

    if low_coverage_clips:
        print(f"\n[ALERT] {len(low_coverage_clips)} clip(s) with < 50% hand detection coverage:")
        for c in low_coverage_clips:
            print(f"  - {c['filename']} ({c['label']}): {c['coverage_ratio']*100:.1f}% coverage ({c['frames_with_hands']}/16 frames)")
    else:
        print("\n[INFO] All clips met the >= 50% hand detection threshold.")
    print("=" * 65)


def evaluate_pipelines(
    X: np.ndarray,
    y: np.ndarray,
    clip_metadata: List[Dict[str, Any]],
    class_names: List[str],
    cv_folds: int = 5,
) -> Tuple[Dict[str, Any], Pipeline]:
    """
    Evaluates multi-class classification pipelines using dynamic Stratified K-Fold CV:
    1) SVM_RBF: StandardScaler + SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True)
    2) SVM_RBF_PCA: StandardScaler + PCA + SVC
    3) Baseline_LogReg: StandardScaler + LogisticRegression(class_weight='balanced')
    """
    class_counts = Counter(y)
    min_class_samples = min(class_counts.values())
    effective_cv_folds = min(cv_folds, min_class_samples)
    effective_cv_folds = max(2, effective_cv_folds)

    print("\n" + "=" * 65)
    print(f"DYNAMIC MULTI-CLASS EVALUATION ({effective_cv_folds}-Fold Stratified CV, {len(class_names)} Classes)")
    print("=" * 65)

    print("[NOTE] Grouped evaluation disclosure:")
    print("  Cross-validation splits are conducted at the video clip level using StratifiedKFold.")
    print("  Cross-session GroupKFold is infeasible when sessions do not equally cover all classes.")
    print("  CV measures consistency across recorded gestures; generalization to unseen signers requires diverse signers.\n")

    skf = StratifiedKFold(n_splits=effective_cv_folds, shuffle=True, random_state=42)

    n_samples = len(y)
    pca_dim = min(15, n_samples - effective_cv_folds, len(class_names) * 3)
    pca_dim = max(2, pca_dim)

    pipelines = {
        "SVM_RBF": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        class_weight="balanced",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "SVM_RBF_PCA": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=pca_dim, random_state=42)),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        class_weight="balanced",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Baseline_LogReg": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0, class_weight="balanced", random_state=42, max_iter=1000
                    ),
                ),
            ]
        ),
    }

    results = {}
    best_pipe_name = "SVM_RBF"
    best_macro_f1 = -1.0

    for name, pipe in pipelines.items():
        oof_preds = np.zeros(len(y), dtype=int)
        fold_accuracies = []
        fold_macro_f1s = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_val)
            oof_preds[val_idx] = preds

            fold_acc = accuracy_score(y_val, preds)
            fold_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
            fold_accuracies.append(fold_acc)
            fold_macro_f1s.append(fold_f1)

        overall_acc = accuracy_score(y, oof_preds)
        overall_macro_f1 = f1_score(y, oof_preds, average="macro", zero_division=0)
        precision_arr, recall_arr, f1_arr, _ = precision_recall_fscore_support(
            y, oof_preds, average=None, zero_division=0, labels=list(range(len(class_names)))
        )
        cm = confusion_matrix(y, oof_preds, labels=list(range(len(class_names))))

        results[name] = {
            "overall_accuracy": float(overall_acc),
            "overall_macro_f1": float(overall_macro_f1),
            "fold_accuracies": [float(a) for a in fold_accuracies],
            "fold_macro_f1s": [float(f) for f in fold_macro_f1s],
            "mean_fold_accuracy": float(np.mean(fold_accuracies)),
            "std_fold_accuracy": float(np.std(fold_accuracies)),
            "per_class": {
                class_names[i]: {
                    "precision": float(precision_arr[i]),
                    "recall": float(recall_arr[i]),
                    "f1": float(f1_arr[i]),
                }
                for i in range(len(class_names))
            },
            "confusion_matrix": cm.tolist(),
        }

        print(f"Pipeline: {name}")
        print(f"  - Overall Accuracy: {overall_acc*100:.2f}%")
        print(f"  - Overall Macro-F1: {overall_macro_f1:.4f}")
        print(f"  - Folds Accuracy:   {np.mean(fold_accuracies)*100:.2f}% ± {np.std(fold_accuracies)*100:.2f}%")
        print("  - Per-Class Breakdown:")
        for i, c in enumerate(class_names):
            print(f"      {c:16s}: Precision={precision_arr[i]:.4f}, Recall={recall_arr[i]:.4f}, F1={f1_arr[i]:.4f}")
        print("  - Confusion Matrix (rows: True, cols: Pred):")
        print(f"      {cm.tolist()}\n")

        if overall_macro_f1 > best_macro_f1:
            best_macro_f1 = overall_macro_f1
            best_pipe_name = name

    # Prioritize SVM_RBF if within 0.005 macro-F1
    if results.get("SVM_RBF", {}).get("overall_macro_f1", 0) >= (best_macro_f1 - 0.005):
        best_pipe_name = "SVM_RBF"
        best_macro_f1 = results["SVM_RBF"]["overall_macro_f1"]

    print(f"[INFO] Selected best pipeline: '{best_pipe_name}' (Macro-F1: {best_macro_f1:.4f})")

    # Fit selected pipeline on all clips
    final_pipeline = pipelines[best_pipe_name]
    final_pipeline.fit(X, y)
    print(f"[INFO] Fitted final '{best_pipe_name}' on all {len(y)} clips across {len(class_names)} classes.")

    return {
        "best_pipeline": best_pipe_name,
        "evaluation_summary": results,
    }, final_pipeline


def save_artifacts_atomically(
    artifacts_dir: str,
    pipeline: Pipeline,
    eval_results: Dict[str, Any],
    clip_metadata: List[Dict[str, Any]],
    class_names: List[str],
    task_model_path: str,
):
    """
    Atomically saves model artifacts and creates backups of previous models.
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    model_path = os.path.join(artifacts_dir, "sign_classifier.joblib")
    metadata_path = os.path.join(artifacts_dir, "model_metadata.json")
    report_path = os.path.join(artifacts_dir, "training_report.json")

    # Preserve backup of existing working artifacts before overwriting
    if os.path.isfile(model_path):
        backup_model = model_path + ".bak"
        shutil.copy2(model_path, backup_model)
        print(f"[INFO] Preserved backup of previous model at: {backup_model}")
    if os.path.isfile(metadata_path):
        backup_meta = metadata_path + ".bak"
        shutil.copy2(metadata_path, backup_meta)
        print(f"[INFO] Preserved backup of previous metadata at: {backup_meta}")

    feature_spec = get_feature_config()

    # 1. Save Pipeline
    with tempfile.NamedTemporaryFile(dir=artifacts_dir, delete=False, suffix=".joblib") as tmp_f:
        tmp_model_path = tmp_f.name
    joblib.dump(pipeline, tmp_model_path)
    os.replace(tmp_model_path, model_path)
    print(f"[INFO] Saved model pipeline atomically to: {model_path}")

    # 2. Build metadata
    label_to_id = {name: idx for idx, name in enumerate(class_names)}
    id_to_label = {str(idx): name for idx, name in enumerate(class_names)}

    model_metadata = {
        "created_at": datetime.now().isoformat(),
        "task": f"Indian Sign Language {len(class_names)}-Class Classification",
        "num_classes": len(class_names),
        "classes": class_names,
        "class_to_id": label_to_id,
        "id_to_class": id_to_label,
        "feature_specification": feature_spec,
        "sequence_length": DEFAULT_NUM_FRAMES,
        "frame_feature_dim": FEATURE_DIM_PER_FRAME,
        "total_feature_dim": TOTAL_CLIP_FEATURE_DIM,
        "mirroring_convention": "Webcam frame horizontally flipped (cv2.flip(frame, 1)) before extraction",
        "task_model_path": task_model_path,
        "task_model_filename": os.path.basename(task_model_path),
        "best_pipeline": eval_results["best_pipeline"],
        "package_versions": {
            "sklearn": sklearn.__version__,
            "joblib": joblib.__version__,
            "numpy": np.__version__,
        },
        "known_limitations": [
            f"Trained on {len(clip_metadata)} clips across {len(class_names)} classes.",
            "Cross-validation reflects recording consistency rather than unseen-signer generalization.",
            "Closed-set classification: unfamiliar gestures may be assigned to one of the known classes.",
        ],
    }

    with tempfile.NamedTemporaryFile(dir=artifacts_dir, delete=False, mode="w", encoding="utf-8", suffix=".json") as tmp_f:
        tmp_meta_path = tmp_f.name
        json.dump(model_metadata, tmp_f, indent=2)
    os.replace(tmp_meta_path, metadata_path)
    print(f"[INFO] Saved metadata atomically to: {metadata_path}")

    # 3. Build training report
    report_data = {
        "training_timestamp": datetime.now().isoformat(),
        "num_classes": len(class_names),
        "classes": class_names,
        "dataset_summary": {
            "total_clips": len(clip_metadata),
            "per_class_counts": dict(Counter(m["label"] for m in clip_metadata)),
        },
        "evaluation": eval_results["evaluation_summary"],
        "selected_pipeline": eval_results["best_pipeline"],
    }

    with tempfile.NamedTemporaryFile(dir=artifacts_dir, delete=False, mode="w", encoding="utf-8", suffix=".json") as tmp_f:
        tmp_rep_path = tmp_f.name
        json.dump(report_data, tmp_f, indent=2)
    os.replace(tmp_rep_path, report_path)
    print(f"[INFO] Saved training report atomically to: {report_path}")

    return model_path, metadata_path, report_path


def verify_saved_artifact(model_path: str, metadata_path: str, test_sample_vector: np.ndarray):
    """
    Verification check: Reloads the artifact from disk and runs inference.
    """
    print("\n" + "=" * 65)
    print("SAVED ARTIFACT RELOAD & PREDICTION VERIFICATION")
    print("=" * 65)

    assert os.path.isfile(model_path), f"Missing model file: {model_path}"
    assert os.path.isfile(metadata_path), f"Missing metadata file: {metadata_path}"

    loaded_pipe = joblib.load(model_path)
    print(f"[PASS] Successfully reloaded pipeline: {loaded_pipe}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print(f"[PASS] Successfully loaded metadata with {len(meta['classes'])} classes: {meta['classes']}")

    expected_dim = meta["total_feature_dim"]
    assert test_sample_vector.shape == (expected_dim,), f"Dimension mismatch: {test_sample_vector.shape} != ({expected_dim},)"

    reshaped_input = test_sample_vector.reshape(1, -1)
    pred_idx = loaded_pipe.predict(reshaped_input)[0]
    pred_probs = loaded_pipe.predict_proba(reshaped_input)[0]
    pred_label = meta["id_to_class"][str(pred_idx)]

    print("[PASS] Prediction smoke test result:")
    print(f"  - Predicted Label: {pred_label} (ID: {pred_idx})")
    print(f"  - Probabilities:   {dict(zip(meta['classes'], [round(float(p), 4) for p in pred_probs]))}")
    assert np.isclose(np.sum(pred_probs), 1.0), "Probabilities do not sum to 1.0!"
    print("[PASS] Artifact verification completed successfully.")
    print("=" * 65)


def main():
    args = parse_args()
    print("=================================================================")
    print("SignVision — Dynamic Multi-Class ISL Training Pipeline")
    print("=================================================================")

    # Stage 1: Dynamic dataset discovery
    discovered_files, metadata_map, sessions_with_both = inspect_dataset(
        args.data_root, args.metadata_csv
    )

    # Stage 2: Feature extraction
    X, y, clip_metadata, class_names = extract_features_dataset(
        discovered_files=discovered_files,
        metadata_map=metadata_map,
        task_model_path=args.task_model,
        cache_path=args.cache_features,
        force_reextract=args.force_reextract,
    )

    # Detection coverage analysis
    analyze_hand_coverage(clip_metadata, class_names)

    # Stage 3 & 4: Evaluate classifiers via CV
    eval_results, final_pipeline = evaluate_pipelines(
        X=X,
        y=y,
        clip_metadata=clip_metadata,
        class_names=class_names,
        cv_folds=args.cv_folds,
    )

    # Stage 5: Save complete artifacts atomically with backups
    model_path, metadata_path, report_path = save_artifacts_atomically(
        artifacts_dir=args.artifacts_dir,
        pipeline=final_pipeline,
        eval_results=eval_results,
        clip_metadata=clip_metadata,
        class_names=class_names,
        task_model_path=args.task_model,
    )

    # Stage 6: Reload and prediction verification
    verify_saved_artifact(model_path, metadata_path, X[0])

    print("\n[SUCCESS] Multi-class training pipeline execution completed.")


if __name__ == "__main__":
    main()
