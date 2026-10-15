#!/usr/bin/env python3
"""
SignVision Multi-Sign Video Recorder (`record_videos.py`)

Interactive data collection tool for Indian Sign Language (ISL) recognition.
Supports dynamic multi-class collection for 10+ signs:
- On-screen control panel with clickable buttons and keyboard input.
- Click [NEW SIGN] to create or switch to any sign name.
- Click [RECORD EXAMPLE] or press [SPACE] to record a 3-second example clip.
- Brief countdown before capture with visual recording indicator.
- Automatically creates and manages class directories under data/raw/<SIGN_NAME>/.
- Updates clip counts dynamically and records full session metadata in data/metadata.csv.
"""

import os
import sys
import time
import uuid
import csv
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.csv")

RECORD_DURATION_SEC = 3.0
TARGET_FPS = 30.0

# UI Geometry Constants
BUTTON_NEW_SIGN = {"x": 16, "y": 70, "w": 120, "h": 32}
INPUT_BOX = {"x": 146, "y": 70, "w": 280, "h": 32}
BUTTON_RECORD = {"x": 436, "y": 70, "w": 190, "h": 32}


def ensure_directories():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)


def init_metadata_csv():
    ensure_directories()
    if not os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "session_id",
                    "timestamp",
                    "label",
                    "filename",
                    "filepath",
                    "duration_sec",
                    "frame_count",
                    "fps",
                ])
        except Exception as e:
            print(f"[ERROR] Failed to initialize metadata CSV: {e}", file=sys.stderr)


def append_metadata(
    session_id: str,
    label: str,
    filename: str,
    filepath: str,
    duration: float,
    frame_count: int,
    fps: float,
):
    try:
        with open(METADATA_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                session_id,
                datetime.now().isoformat(),
                label,
                filename,
                filepath,
                f"{duration:.2f}",
                frame_count,
                f"{fps:.2f}",
            ])
    except Exception as e:
        print(f"[ERROR] Failed to write metadata to {METADATA_FILE}: {e}", file=sys.stderr)


def discover_all_classes() -> List[str]:
    """Scans data/raw/ for all existing class directories."""
    ensure_directories()
    classes = []
    for entry in sorted(os.listdir(RAW_DIR)):
        full_path = os.path.join(RAW_DIR, entry)
        if os.path.isdir(full_path) and not entry.startswith("."):
            classes.append(entry)
    return classes


def get_all_clip_counts() -> Dict[str, int]:
    """Counts video clips in each class directory."""
    classes = discover_all_classes()
    counts = {}
    for cls in classes:
        cls_dir = os.path.join(RAW_DIR, cls)
        files = [
            f
            for f in os.listdir(cls_dir)
            if f.lower().endswith((".mp4", ".avi", ".mov"))
        ]
        counts[cls] = len(files)
    return counts


def validate_and_normalize_sign_name(raw_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Normalizes sign name to UPPERCASE_WITH_UNDERSCORES.
    Rejects empty, path-traversal, or illegal characters.
    """
    name = raw_name.strip().replace(" ", "_").replace("-", "_").upper()
    while "__" in name:
        name = name.replace("__", "_")
    name = name.strip("_")

    if not name:
        return None, "Sign name cannot be empty."
    if not re.match(r"^[A-Z0-9_]+$", name):
        return None, "Only letters, numbers, and underscores allowed."
    if name in (".", "..") or "/" in name or "\\" in name:
        return None, "Invalid sign name."
    return name, None


def open_camera(preferred_idx: int = 0):
    for idx in (preferred_idx, 0, 1, 2):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
                return cap, idx
            cap.release()
    return None, -1


def is_point_inside(pt: Tuple[int, int], box: Dict[str, int]) -> bool:
    x, y = pt
    return (
        box["x"] <= x <= box["x"] + box["w"]
        and box["y"] <= y <= box["y"] + box["h"]
    )


def draw_control_panel(
    frame: np.ndarray,
    active_sign: str,
    clip_counts: Dict[str, int],
    typing_active: bool,
    input_text: str,
    feedback_msg: str,
    feedback_is_error: bool,
    state: str = "IDLE",
    record_elapsed: float = 0.0,
) -> np.ndarray:
    """
    Renders top interactive control panel, buttons, text box, counts, and bottom instructions.
    """
    h, w, _ = frame.shape
    overlay = frame.copy()

    # Top panel background
    top_bar_h = 152
    cv2.rectangle(overlay, (0, 0), (w, top_bar_h), (20, 20, 20), -1)

    # Bottom bar background
    bottom_bar_h = 56
    cv2.rectangle(overlay, (0, h - bottom_bar_h), (w, h), (20, 20, 20), -1)

    alpha = 0.85
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Title & stats
    title = "SignVision Multi-Sign Video Recorder"
    cv2.putText(frame, title, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    total_clips = sum(clip_counts.values())
    active_count = clip_counts.get(active_sign, 0)
    num_classes = len(clip_counts)
    stats_str = f"Active: {active_sign} ({active_count} clips) | Total: {total_clips} clips across {num_classes} signs"
    cv2.putText(frame, stats_str, (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

    # --- BUTTON 1: [NEW SIGN] ---
    btn_ns = BUTTON_NEW_SIGN
    ns_color = (0, 165, 255) if typing_active else (60, 60, 60)
    cv2.rectangle(frame, (btn_ns["x"], btn_ns["y"]), (btn_ns["x"] + btn_ns["w"], btn_ns["y"] + btn_ns["h"]), ns_color, -1)
    cv2.rectangle(frame, (btn_ns["x"], btn_ns["y"]), (btn_ns["x"] + btn_ns["w"], btn_ns["y"] + btn_ns["h"]), (200, 200, 200), 1)
    cv2.putText(frame, "+ NEW SIGN", (btn_ns["x"] + 12, btn_ns["y"] + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # --- INPUT BOX: Sign Name ---
    ib = INPUT_BOX
    ib_bg = (35, 35, 35) if typing_active else (25, 25, 25)
    ib_border = (0, 255, 255) if typing_active else (80, 80, 80)
    cv2.rectangle(frame, (ib["x"], ib["y"]), (ib["x"] + ib["w"], ib["y"] + ib["h"]), ib_bg, -1)
    cv2.rectangle(frame, (ib["x"], ib["y"]), (ib["x"] + ib["w"], ib["y"] + ib["h"]), ib_border, 1)

    cursor = "_" if (typing_active and int(time.time() * 2) % 2 == 0) else ""
    display_input = (input_text if typing_active else active_sign) + cursor
    text_color = (0, 255, 255) if typing_active else (240, 240, 240)
    cv2.putText(frame, display_input[:20], (ib["x"] + 10, ib["y"] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)

    # --- BUTTON 2: [RECORD EXAMPLE] ---
    btn_rec = BUTTON_RECORD
    if state == "RECORDING":
        rec_bg = (0, 0, 220)
        rec_text = f"REC ({record_elapsed:.1f}s/3.0s)"
    else:
        rec_bg = (0, 180, 0)
        rec_text = "● RECORD EXAMPLE (3s)"

    cv2.rectangle(frame, (btn_rec["x"], btn_rec["y"]), (btn_rec["x"] + btn_rec["w"], btn_rec["y"] + btn_rec["h"]), rec_bg, -1)
    cv2.rectangle(frame, (btn_rec["x"], btn_rec["y"]), (btn_rec["x"] + btn_rec["w"], btn_rec["y"] + btn_rec["h"]), (255, 255, 255), 1)
    cv2.putText(frame, rec_text, (btn_rec["x"] + 10, btn_rec["y"] + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # Feedback / status line (row 4)
    fb_color = (0, 100, 255) if feedback_is_error else (0, 255, 150)
    cv2.putText(frame, feedback_msg, (16, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.50, fb_color, 1, cv2.LINE_AA)

    # Center recording indicator
    if state == "RECORDING":
        cv2.circle(frame, (w - 40, 32), 14, (0, 0, 255), -1)
        cv2.putText(frame, f"RECORDING '{active_sign}'", (w // 2 - 150, h // 2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    # Bottom Instructions Bar
    instructions = "Click [RECORD EXAMPLE] or Press [SPACE] to Record  |  Click [NEW SIGN] to change  |  [Q/ESC] Quit"
    cv2.putText(frame, instructions, (16, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


def execute_single_recording(
    cap: cv2.VideoCapture,
    label: str,
    session_id: str,
    window_name: str,
    active_sign: str,
    clip_counts: Dict[str, int],
) -> Tuple[bool, str]:
    """
    Executes exactly one 3-second recording instantly:
    1. Direct 3-second recording.
    2. Saves video under data/raw/<label>/.
    3. Appends to metadata.csv.
    """
    target_dir = os.path.join(RAW_DIR, label)
    os.makedirs(target_dir, exist_ok=True)

    # Direct Recording Phase (3.0 seconds)
    recorded_frames = []
    rec_start = time.time()
    while True:
        elapsed = time.time() - rec_start
        if elapsed >= RECORD_DURATION_SEC:
            break

        ret, frame = cap.read()
        if not ret or frame is None:
            return False, "Camera frame drop during recording."

        frame = cv2.flip(frame, 1)
        recorded_frames.append(frame.copy())

        hud = draw_control_panel(
            frame=frame.copy(),
            active_sign=active_sign,
            clip_counts=clip_counts,
            typing_active=False,
            input_text="",
            feedback_msg=f"Recording '{label}'... Keep signing!",
            feedback_is_error=False,
            state="RECORDING",
            record_elapsed=elapsed,
        )
        cv2.imshow(window_name, hud)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            return False, "Recording aborted by user."

    actual_duration = time.time() - rec_start
    num_frames = len(recorded_frames)
    if num_frames == 0:
        return False, "No frames captured."

    fps = num_frames / actual_duration if actual_duration > 0 else TARGET_FPS
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_suffix = uuid.uuid4().hex[:6]
    filename = f"{label}_{timestamp_str}_{unique_suffix}.mp4"
    save_path = os.path.join(target_dir, filename)

    h, w, _ = recorded_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    if not out.isOpened():
        # Fallback to AVI / XVID
        filename = f"{label}_{timestamp_str}_{unique_suffix}.avi"
        save_path = os.path.join(target_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    if not out.isOpened():
        return False, f"Could not create video file at {save_path}"

    for f in recorded_frames:
        out.write(f)
    out.release()

    append_metadata(session_id, label, filename, save_path, actual_duration, num_frames, fps)
    print(f"[SUCCESS] Saved: {save_path} ({num_frames} frames, {actual_duration:.2f}s, {fps:.1f} FPS)")
    return True, f"Saved '{filename}' ({num_frames} frames)"


# Global mouse event state
mouse_action = None


def on_mouse_click(event, x, y, flags, param):
    global mouse_action
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_action = (x, y)


def main():
    global mouse_action
    session_id = uuid.uuid4().hex[:8]
    print(f"[INFO] Launching SignVision Multi-Sign Video Recorder (Session: {session_id})...")
    ensure_directories()
    init_metadata_csv()

    # Discover initial classes
    existing_classes = discover_all_classes()
    if not existing_classes:
        active_sign = "GOOD_MORNING"
        os.makedirs(os.path.join(RAW_DIR, active_sign), exist_ok=True)
    else:
        active_sign = existing_classes[0]

    clip_counts = get_all_clip_counts()
    print(f"[INFO] Initial classes discovered: {list(clip_counts.keys())}")
    print(f"[INFO] Active sign: '{active_sign}' ({clip_counts.get(active_sign, 0)} clips)")

    cap, camera_idx = open_camera()
    if cap is None:
        print("[ERROR] Unable to access camera. Please check your webcam connection or permissions.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Camera opened successfully on device index {camera_idx}.")
    window_name = "SignVision Multi-Sign Video Recorder"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse_click)

    typing_active = False
    input_text = ""
    feedback_msg = f"Ready. Click [RECORD EXAMPLE] or press [SPACE] to record '{active_sign}'."
    feedback_is_error = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[ERROR] Camera frame read failure.", file=sys.stderr)
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)

            # Process mouse clicks
            if mouse_action is not None:
                click_pt = mouse_action
                mouse_action = None

                if is_point_inside(click_pt, BUTTON_NEW_SIGN) or is_point_inside(click_pt, INPUT_BOX):
                    typing_active = True
                    input_text = ""
                    feedback_msg = "Type new sign name, then press [ENTER] to confirm or [ESC] to cancel."
                    feedback_is_error = False
                    print("[INFO] Text input mode activated.")

                elif is_point_inside(click_pt, BUTTON_RECORD):
                    if typing_active:
                        # Auto-confirm input first if user was typing
                        norm_name, err = validate_and_normalize_sign_name(input_text)
                        if norm_name:
                            active_sign = norm_name
                            os.makedirs(os.path.join(RAW_DIR, active_sign), exist_ok=True)
                            clip_counts = get_all_clip_counts()
                            typing_active = False
                    # Start record
                    print(f"[ACTION] Recording triggered for sign: '{active_sign}'")
                    success, msg = execute_single_recording(
                        cap=cap,
                        label=active_sign,
                        session_id=session_id,
                        window_name=window_name,
                        active_sign=active_sign,
                        clip_counts=clip_counts,
                    )
                    clip_counts = get_all_clip_counts()
                    feedback_msg = msg
                    feedback_is_error = not success

            # Render HUD
            hud_frame = draw_control_panel(
                frame=frame.copy(),
                active_sign=active_sign,
                clip_counts=clip_counts,
                typing_active=typing_active,
                input_text=input_text,
                feedback_msg=feedback_msg,
                feedback_is_error=feedback_is_error,
                state="IDLE",
            )
            cv2.imshow(window_name, hud_frame)

            # Keyboard Events
            key = cv2.waitKey(20) & 0xFF

            if typing_active:
                if key in (13, 10):  # ENTER key
                    norm_name, err = validate_and_normalize_sign_name(input_text)
                    if err:
                        feedback_msg = f"Error: {err}"
                        feedback_is_error = True
                        print(f"[WARN] {err}")
                    else:
                        active_sign = norm_name
                        os.makedirs(os.path.join(RAW_DIR, active_sign), exist_ok=True)
                        clip_counts = get_all_clip_counts()
                        typing_active = False
                        input_text = ""
                        feedback_msg = f"Active sign set to: '{active_sign}'. Ready to record."
                        feedback_is_error = False
                        print(f"[INFO] Switched active sign to: '{active_sign}'")

                elif key == 27:  # ESC cancels typing
                    typing_active = False
                    input_text = ""
                    feedback_msg = f"Canceled. Active sign remains: '{active_sign}'."
                    feedback_is_error = False

                elif key in (8, 127):  # Backspace
                    input_text = input_text[:-1]

                elif 32 <= key <= 126:  # Printable character
                    char = chr(key).upper()
                    if char == " ":
                        char = "_"
                    if len(input_text) < 25 and (char.isalnum() or char == "_"):
                        input_text += char

            else:
                # Normal mode
                if key in (ord("q"), ord("Q"), 27):
                    print("[INFO] Quit requested by user.")
                    break

                elif key in (ord("n"), ord("N")):
                    typing_active = True
                    input_text = ""
                    feedback_msg = "Type sign name, then press [ENTER] to confirm or [ESC] to cancel."
                    feedback_is_error = False

                elif key in (32, ord("r"), ord("R")):  # SPACE or 'R'
                    print(f"[ACTION] Recording triggered for sign: '{active_sign}'")
                    success, msg = execute_single_recording(
                        cap=cap,
                        label=active_sign,
                        session_id=session_id,
                        window_name=window_name,
                        active_sign=active_sign,
                        clip_counts=clip_counts,
                    )
                    clip_counts = get_all_clip_counts()
                    feedback_msg = msg
                    feedback_is_error = not success

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Video Recorder cleanly shut down.")


if __name__ == "__main__":
    main()
