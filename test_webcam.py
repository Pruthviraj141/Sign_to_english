#!/usr/bin/env python3
"""
SignVision Dynamic Multi-Class Webcam Testing Script (test_webcam.py)

Tests live Indian Sign Language (ISL) recognition on webcam for any number of trained signs
(e.g., GOOD_MORNING, GOOD_NIGHT, THANK_YOU, HELP, WATER, etc.):
- Dynamically loads all supported classes from artifacts/model_metadata.json.
- Live preview with hand skeleton overlay so you can position your hands.
- Press [SPACE] to record a 3-second test clip.
- Automatically samples 16 frames uniformly across the clip, extracts features via
  src.landmark_features, and runs the multi-class classifier.
- Displays predicted sign, top candidate probabilities, and hand coverage.
- Press [SPACE] again to test another sign without restarting.

Controls:
    [SPACE]   Record 3.0s clip and predict
    [H]       Toggle hand skeleton visualization on/off
    [S]       Save diagnostic screenshot to screenshots/
    [C]       Clear last prediction result card
    [Q / ESC] Quit
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import joblib

from src.landmark_features import (
    HandLandmarkerExtractor,
    HAND_CONNECTIONS,
    FINGERTIP_INDICES,
    WRIST_INDEX,
    DEFAULT_NUM_FRAMES,
    FEATURE_DIM_PER_FRAME,
    TOTAL_CLIP_FEATURE_DIM,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SignVision ISL Dynamic Multi-Class Webcam Testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Recording duration in seconds for test clip",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Confidence probability threshold for accepting a prediction",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="artifacts/sign_classifier.joblib",
        help="Path to trained classifier pipeline",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="artifacts/model_metadata.json",
        help="Path to model metadata JSON",
    )
    parser.add_argument(
        "--task-model",
        type=str,
        default="models/hand_landmarker.task",
        help="Path to MediaPipe HandLandmarker task model",
    )
    parser.add_argument(
        "--min-hand-coverage",
        type=float,
        default=0.50,
        help="Minimum fraction of frames in the 16-frame window that must contain detected hands",
    )
    return parser.parse_args()


def load_artifacts(
    model_path: str, metadata_path: str, task_model_path: str
) -> Tuple[Any, Dict[str, Any]]:
    """
    Validates file existence and loads the trained pipeline and metadata.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Trained model not found at: {model_path}\n"
            "Please run `python3 train_model.py` first."
        )
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"Model metadata not found at: {metadata_path}\n"
            "Please run `python3 train_model.py` first."
        )
    if not os.path.isfile(task_model_path):
        raise FileNotFoundError(
            f"MediaPipe task model not found at: {task_model_path}\n"
            "Ensure models/hand_landmarker.task exists."
        )

    pipeline = joblib.load(model_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    expected_dim = metadata.get("total_feature_dim", TOTAL_CLIP_FEATURE_DIM)
    if expected_dim != TOTAL_CLIP_FEATURE_DIM:
        raise ValueError(
            f"Metadata total_feature_dim ({expected_dim}) does not match current code ({TOTAL_CLIP_FEATURE_DIM})!"
        )

    classes = metadata.get("classes", [])
    print(f"[INFO] Loaded classifier: {type(pipeline).__name__}")
    print(f"[INFO] Trained on {len(classes)} classes: {classes}")
    print(f"[INFO] Sequence length: {metadata.get('sequence_length', DEFAULT_NUM_FRAMES)} frames")
    print(f"[INFO] Expected feature vector dimension: {expected_dim}")

    return pipeline, metadata


def open_camera(preferred_idx: int = 0):
    for idx in (preferred_idx, 0, 1, 2):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.set(cv2.CAP_PROP_FPS, 30.0)
                return cap, idx
            cap.release()
    return None, -1


def draw_hand_skeleton(frame: np.ndarray, hand_landmarks_list: List[Any]):
    h, w, _ = frame.shape
    for landmarks in hand_landmarks_list:
        pts = []
        for lm in landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))
            pts.append((px, py))

        for idx1, idx2 in HAND_CONNECTIONS:
            if idx1 < len(pts) and idx2 < len(pts):
                cv2.line(frame, pts[idx1], pts[idx2], (0, 230, 230), 2, cv2.LINE_AA)

        for i, (px, py) in enumerate(pts):
            if i == WRIST_INDEX:
                cv2.circle(frame, (px, py), 6, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 8, (20, 20, 20), 1, cv2.LINE_AA)
            elif i in FINGERTIP_INDICES:
                cv2.circle(frame, (px, py), 6, (255, 180, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 8, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (px, py), 4, (0, 220, 100), -1, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    state: str,
    record_progress: float,
    duration: float,
    last_result: Optional[Dict[str, Any]],
    classes: List[str],
    hands_detected: int,
    handedness: List[str],
    fps: float,
    show_landmarks: bool,
    screenshot_notice: str = "",
) -> np.ndarray:
    """
    Renders top status bar with dynamic multi-class predictions, probability bars,
    hand tracking diagnostics, and bottom control prompts.
    """
    h, w, _ = frame.shape
    overlay = frame.copy()

    # Dynamic top banner height based on number of top classes to display
    top_bar_h = 145
    cv2.rectangle(overlay, (0, 0), (w, top_bar_h), (18, 18, 18), -1)

    # Bottom bar
    bottom_bar_h = 52
    cv2.rectangle(overlay, (0, h - bottom_bar_h), (w, h), (18, 18, 18), -1)

    alpha = 0.82
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Title & FPS
    title = f"SignVision Multi-Class ISL Recognition ({len(classes)} Signs)"
    cv2.putText(frame, title, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    fps_text = f"Live FPS: {fps:4.1f}"
    cv2.putText(frame, fps_text, (w - 180, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    if state == "RECORDING":
        cv2.circle(frame, (30, 65), 14, (0, 0, 255), -1)
        cv2.putText(frame, f"RECORDING SIGN... ({record_progress:.1f}s / {duration:.1f}s)", (56, 73),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 0, 255), 2, cv2.LINE_AA)

        # Progress bar
        bar_w = int(w * 0.7)
        bar_x = (w - bar_w) // 2
        bar_y = 105
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 14), (50, 50, 50), -1)
        fill_w = int(bar_w * min(1.0, record_progress / max(0.1, duration)))
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + 14), (0, 0, 255), -1)

    elif state == "PREDICTING":
        cv2.putText(frame, "ANALYZING 16-FRAME SEQUENCE...", (16, 73),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 200, 255), 2, cv2.LINE_AA)

    else:
        # IDLE
        if last_result is not None:
            pred_label = last_result["label"]
            confidence = last_result["confidence"]
            sorted_probs = last_result.get("sorted_probabilities", [])
            coverage = last_result["coverage"]

            if pred_label == "UNCERTAIN":
                res_color = (0, 165, 255)  # Orange
                banner_text = f"RESULT: UNCERTAIN ({last_result['status_msg']})"
            else:
                res_color = (0, 255, 100)  # Green
                banner_text = f"PREDICTED SIGN: {pred_label} ({confidence*100:.1f}%)"

            cv2.putText(frame, banner_text, (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.82, res_color, 2, cv2.LINE_AA)

            # Display top 3 candidate class probabilities dynamically
            top_candidates = sorted_probs[:3]
            prob_y = 88
            for cls_name, p_val in top_candidates:
                disp_name = (cls_name[:18] + "..") if len(cls_name) > 19 else cls_name
                cv2.putText(frame, f"{disp_name:19s}: {p_val*100:5.1f}%", (16, prob_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)
                # Mini bar
                bx = 230
                bw = 90
                cv2.rectangle(frame, (bx, prob_y - 10), (bx + bw, prob_y), (60, 60, 60), 1)
                fill_p = int(bw * p_val)
                if fill_p > 0:
                    cv2.rectangle(frame, (bx + 1, prob_y - 9), (bx + fill_p, prob_y - 1), (0, 220, 100), -1)
                prob_y += 18

            # Hand coverage of last clip
            cv2.putText(frame, f"Clip Hand Coverage: {coverage*100:.1f}%", (w - 360, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Status: {last_result['status_msg']}", (w - 360, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)

        else:
            ready_prompt = "READY — Perform sign & press [SPACE] to record & predict"
            cv2.putText(frame, ready_prompt, (16, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 150), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Trained vocabulary ({len(classes)} signs): {', '.join(classes[:6])}{'...' if len(classes) > 6 else ''}",
                        (16, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

    # Real-time Hands Detection status (top right)
    if state != "RECORDING":
        hands_str = f"Live Hands: {hands_detected} ({', '.join(handedness) if handedness else 'None'})"
        cv2.putText(frame, hands_str, (w - 360, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

    # Screenshot notice banner
    if screenshot_notice:
        cv2.rectangle(frame, (w // 2 - 200, h // 2 - 25), (w // 2 + 200, h // 2 + 25), (20, 20, 20), -1)
        cv2.rectangle(frame, (w // 2 - 200, h // 2 - 25), (w // 2 + 200, h // 2 + 25), (0, 255, 0), 2)
        cv2.putText(frame, screenshot_notice, (w // 2 - 180, h // 2 + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

    # Bottom instructions bar
    instructions = f"[SPACE] Record {duration:.1f}s Sign & Predict   |   [H] Toggle Skeleton   |   [S] Screenshot   |   [Q/ESC] Quit"
    cv2.putText(frame, instructions, (16, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

    return frame


def process_recorded_clip(
    recorded_frames: List[np.ndarray],
    extractor: HandLandmarkerExtractor,
    pipeline: Any,
    classes: List[str],
    id_to_class: Dict[str, str],
    threshold: float,
    min_hand_coverage: float,
) -> Dict[str, Any]:
    """
    Uniformly samples 16 frames from the recorded clip, extracts landmark features,
    and runs multi-class classification.
    """
    total_frames = len(recorded_frames)
    if total_frames == 0:
        return {
            "label": "UNCERTAIN",
            "confidence": 0.0,
            "probabilities": {c: 0.0 for c in classes},
            "sorted_probabilities": [],
            "coverage": 0.0,
            "status_msg": "No frames captured",
        }

    # Sample 16 frames uniformly
    if total_frames >= DEFAULT_NUM_FRAMES:
        indices = np.linspace(0, total_frames - 1, DEFAULT_NUM_FRAMES, dtype=int)
    else:
        indices = list(range(total_frames))
        while len(indices) < DEFAULT_NUM_FRAMES:
            indices.append(indices[-1])
        indices = np.array(indices[:DEFAULT_NUM_FRAMES], dtype=int)

    sampled_frames = [recorded_frames[idx] for idx in indices]

    frame_vectors = []
    frames_with_hands = 0
    for frame in sampled_frames:
        vec, stats = extractor.extract_frame_landmarks(frame)
        frame_vectors.append(vec)
        if stats.get("hands_detected", 0) > 0:
            frames_with_hands += 1

    clip_vector = np.concatenate(frame_vectors).astype(np.float32)
    coverage = float(frames_with_hands) / float(DEFAULT_NUM_FRAMES)

    # Multi-class Classifier prediction
    prob_array = pipeline.predict_proba(clip_vector.reshape(1, -1))[0]

    # Map probabilities respecting pipeline.classes_
    classifier_classes = getattr(
        pipeline,
        "classes_",
        getattr(pipeline.named_steps.get("classifier", None), "classes_", list(range(len(classes)))),
    )
    prob_dict = {}
    for idx_in_array, class_idx in enumerate(classifier_classes):
        idx_str = str(int(class_idx))
        if idx_str in id_to_class:
            class_name = id_to_class[idx_str]
        elif int(class_idx) < len(classes):
            class_name = classes[int(class_idx)]
        else:
            class_name = f"Class_{class_idx}"
        prob_dict[class_name] = float(prob_array[idx_in_array])

    sorted_probs = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
    top_label, top_prob = sorted_probs[0]

    if coverage < min_hand_coverage:
        final_label = "UNCERTAIN"
        status_msg = f"Low hand coverage ({coverage*100:.0f}%)"
    elif top_prob < threshold:
        final_label = "UNCERTAIN"
        status_msg = f"Low confidence ({top_prob*100:.0f}%)"
    else:
        final_label = top_label
        status_msg = "Confident Prediction"

    return {
        "label": final_label,
        "confidence": top_prob,
        "probabilities": prob_dict,
        "sorted_probabilities": sorted_probs,
        "coverage": coverage,
        "status_msg": status_msg,
        "total_frames": total_frames,
    }


def main():
    args = parse_args()
    print("=================================================================")
    print("SignVision — Dynamic Multi-Class ISL Webcam Testing")
    print("=================================================================")
    print(f"[INFO] Test clip recording duration: {args.duration:.1f} seconds")
    print(f"[INFO] Confidence probability threshold: {args.threshold:.2f}")
    print(f"[INFO] Minimum hand detection coverage: {args.min_hand_coverage * 100:.0f}%")

    try:
        pipeline, metadata = load_artifacts(args.model, args.metadata, args.task_model)
    except Exception as e:
        print(f"[FATAL] Failed to load artifacts: {e}", file=sys.stderr)
        sys.exit(1)

    classes = metadata.get("classes", [])
    id_to_class = metadata.get("id_to_class", {str(i): c for i, c in enumerate(classes)})

    print(f"[INFO] Initializing MediaPipe HandLandmarker from {args.task_model}...")
    try:
        extractor = HandLandmarkerExtractor(model_path=args.task_model, num_hands=2)
    except Exception as e:
        print(f"[FATAL] Failed to initialize MediaPipe detector: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Opening webcam (index {args.camera})...")
    cap, camera_idx = open_camera(args.camera)
    if cap is None:
        print(
            f"[FATAL] Unable to open camera on index {args.camera} or fallbacks. Please verify webcam connection.",
            file=sys.stderr,
        )
        extractor.close()
        sys.exit(1)

    print(f"[INFO] Camera opened successfully on device index {camera_idx}.")
    window_name = f"SignVision Multi-Class ISL Recognition ({len(classes)} Signs)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    os.makedirs("screenshots", exist_ok=True)

    state = "IDLE"
    recorded_frames: List[np.ndarray] = []
    record_start_time = 0.0
    last_result: Optional[Dict[str, Any]] = None

    show_landmarks = True
    screenshot_notice = ""
    screenshot_notice_until = 0.0

    fps_smooth = 30.0
    t_prev_frame = time.perf_counter()

    print("\n[READY] Live preview running.")
    print(" -> Press [SPACE] to record a test clip and predict the sign.")
    print(" -> Press [Q] or [ESC] to quit.\n")

    try:
        while True:
            t_frame_start = time.perf_counter()

            ret, raw_frame = cap.read()
            if not ret or raw_frame is None:
                print("[WARN] Camera frame drop or read error.", file=sys.stderr)
                time.sleep(0.01)
                continue

            frame = cv2.flip(raw_frame, 1)

            # Extract landmarks for real-time visual feedback
            _, stats = extractor.extract_frame_landmarks(frame)
            hands_detected = stats.get("hands_detected", 0)
            handedness = stats.get("handedness", [])

            # State Machine: RECORDING vs IDLE
            record_progress = 0.0
            if state == "RECORDING":
                recorded_frames.append(frame.copy())
                elapsed = time.time() - record_start_time
                record_progress = elapsed

                if elapsed >= args.duration:
                    state = "PREDICTING"
                    hud_analyzing = draw_hud(
                        frame=frame.copy(),
                        state="PREDICTING",
                        record_progress=args.duration,
                        duration=args.duration,
                        last_result=None,
                        classes=classes,
                        hands_detected=hands_detected,
                        handedness=handedness,
                        fps=fps_smooth,
                        show_landmarks=show_landmarks,
                    )
                    cv2.imshow(window_name, hud_analyzing)
                    cv2.waitKey(1)

                    print(f"[INFO] Clip recorded ({len(recorded_frames)} frames). Running inference...")
                    last_result = process_recorded_clip(
                        recorded_frames=recorded_frames,
                        extractor=extractor,
                        pipeline=pipeline,
                        classes=classes,
                        id_to_class=id_to_class,
                        threshold=args.threshold,
                        min_hand_coverage=args.min_hand_coverage,
                    )
                    print(f"[RESULT] Predicted: {last_result['label']} (Confidence: {last_result['confidence']*100:.1f}%) | {last_result['status_msg']}")
                    state = "IDLE"
                    recorded_frames = []

            # Hand skeleton overlay
            if show_landmarks and "hand_landmarks" in stats:
                draw_hand_skeleton(frame, stats["hand_landmarks"])

            # FPS calculation
            t_frame_end = time.perf_counter()
            dt = t_frame_end - t_prev_frame
            t_prev_frame = t_frame_end
            if dt > 0:
                current_fps = 1.0 / dt
                fps_smooth = 0.9 * fps_smooth + 0.1 * current_fps

            if time.time() > screenshot_notice_until:
                screenshot_notice = ""

            hud_frame = draw_hud(
                frame=frame,
                state=state,
                record_progress=record_progress,
                duration=args.duration,
                last_result=last_result,
                classes=classes,
                hands_detected=hands_detected,
                handedness=handedness,
                fps=fps_smooth,
                show_landmarks=show_landmarks,
                screenshot_notice=screenshot_notice,
            )

            cv2.imshow(window_name, hud_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("[INFO] Quit requested by user.")
                break
            elif key == 32:  # SPACE bar
                if state == "IDLE":
                    state = "RECORDING"
                    recorded_frames = [frame.copy()]
                    record_start_time = time.time()
                    print(f"[INFO] Started recording {args.duration:.1f}s test clip...")
            elif key in (ord("h"), ord("H")):
                show_landmarks = not show_landmarks
                print(f"[INFO] Skeleton overlay toggled: {'ON' if show_landmarks else 'OFF'}")
            elif key in (ord("s"), ord("S")):
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = os.path.join("screenshots", f"result_{stamp}.png")
                cv2.imwrite(snap_path, hud_frame)
                screenshot_notice = f"Saved: {snap_path}"
                screenshot_notice_until = time.time() + 2.0
                print(f"[INFO] Saved screenshot: {snap_path}")
            elif key in (ord("c"), ord("C")):
                last_result = None
                print("[INFO] Cleared previous result.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        print("[INFO] Camera and detector cleanly released.")


if __name__ == "__main__":
    main()
