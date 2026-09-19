"""
pancake_detection.py

Live pancake tracker. Opens your webcam (or a video file), finds the pancake,
and shows a window with:
    LEFT  : the camera image with the pancake highlighted (coloured overlay,
            bounding box, center dot, position + size + match score)
    RIGHT : the mask, i.e. what the detector thinks is "pancake colour"

HOW A BLOB IS JUDGED TO BE THE PANCAKE
    1. COLOUR   Pixels must fall inside the HSV range (set by clicking on the
                pancake). This only creates *candidate blobs*.
    2. SHAPE    Every candidate blob gets a "match score" from 0 to 100%:
                    circularity  - how close to a circle/ellipse the outline is
                    solidity     - how few dents/holes (blob area / convex hull)
                    ellipse fit  - how well a fitted ellipse covers the blob
                match = geometric mean of the three. A flat pancake seen by
                the camera is a clean ellipse, so it scores high (90%+), while
                ragged background regions of similar colour score low.
                Blobs cut off by the edge of the frame are penalised, since
                a truncated shape is usually background.
    3. CHOICE   Among blobs above "Min shape %", the best one wins. A blob
                that is bigger and near where the pancake was in the previous
                frame gets a small bonus, so the box doesn't jump around.
                Thin strips of background touching the pancake are trimmed
                off first, so they don't distort its box or score.

SETUP:
    pip install opencv-python numpy

RUN:
    python pancake_detection.py                    # auto-finds your webcam
    python pancake_detection.py --list-cameras     # shows which cameras work
    python pancake_detection.py --source 1         # use camera number 1
    python pancake_detection.py --source video.mp4 # use a recorded video

FIRST THING TO DO: click on the middle of the pancake in the left panel. That
sets the colour range from the pancake itself. Then use the sliders if needed.

CONTROLS (click the video window first so it has focus):
    Left-click on the pancake -> sets the HSV colour range from that spot
    Sliders (HSV Controls)    -> H/S/V colour range, Min area, Cleanup px
                                 (bigger = separates touching blobs better),
                                 Min shape % (how round a blob must be)
    SPACE                     -> pause / resume
    M                         -> show / hide the mask panel
    C                         -> show / hide the rejected candidates (red)
    S                         -> save a screenshot to ./screenshots
    P                         -> print the current detection + settings
    Q or ESC                  -> quit (prints your final HSV settings)

Coordinates are in pixels of the displayed frame (max 640 px wide per panel).
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

WINDOW_MAIN = "Pancake Tracker"
WINDOW_CTRL = "HSV Controls"

PANEL_WIDTH = 640        # camera frames are shrunk to at most this width
MAX_CAMERA_INDEX = 5     # auto-scan tries cameras 0..5
REQUEST_SIZE = (1280, 720)

# Shape / tracking parameters
MAX_AREA_FRACTION = 0.6   # a blob covering more of the frame than this is background
MIN_ASPECT = 0.35         # minor/major axis; thinner than this can't be a pancake
BORDER_PENALTY = 0.6      # score multiplier for blobs cut off by the frame edge
CIRCULARITY_REF = 0.85    # a perfect circle measures ~0.85-0.9 once rasterised
LOST_FRAMES_RESET = 15    # forget the previous position after this many misses
MAX_DRAWN_CANDIDATES = 6  # how many rejected blobs to outline in red

FONT = cv2.FONT_HERSHEY_SIMPLEX
BOX_COLOR = (0, 255, 0)          # green bounding box (BGR)
HIGHLIGHT_COLOR = (0, 255, 255)  # yellow translucent pancake highlight (BGR)
REJECT_COLOR = (80, 80, 255)     # red outline for rejected candidates (BGR)

# name -> (starting value, max value)
# Defaults target a bright yellow/golden pancake. Click on your pancake to
# calibrate for your own lighting.
TRACKBARS = {
    "H low": (18, 179),
    "H high": (34, 179),
    "S low": (100, 255),
    "S high": (255, 255),
    "V low": (130, 255),
    "V high": (255, 255),
    "Min area": (1500, 30000),
    "Cleanup px": (9, 25),
    "Min shape %": (70, 100),
}


# --------------------------------------------------------------------------
# Camera access
# --------------------------------------------------------------------------

def backends_to_try():
    """Windows webcams are flaky, so we try several capture backends."""
    if sys.platform.startswith("win"):
        return [
            ("DirectShow", cv2.CAP_DSHOW),
            ("Media Foundation", cv2.CAP_MSMF),
            ("Default", cv2.CAP_ANY),
        ]
    return [("Default", cv2.CAP_ANY)]


def try_open_camera(index):
    """
    Try to open camera `index` with each backend. A camera only counts as
    working if it actually delivers a real (non-black) frame, because some
    drivers "open" fine but return nothing.

    Returns (cap, backend_name) or (None, None).
    """
    for backend_name, backend in backends_to_try():
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUEST_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_SIZE[1])

        # Webcams often return black/empty frames while auto-exposure warms up
        for _ in range(20):
            ret, frame = cap.read()
            if ret and frame is not None and frame.max() > 0:
                return cap, backend_name
            time.sleep(0.05)

        cap.release()
    return None, None


def list_cameras():
    print("Scanning for cameras (this can take a few seconds)...")
    found = False
    for index in range(MAX_CAMERA_INDEX + 1):
        cap, backend_name = try_open_camera(index)
        if cap is not None:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  Camera {index}: working  ({width}x{height}, {backend_name})")
            cap.release()
            found = True
    if not found:
        print("  No working cameras found.")
        print_camera_help()


def print_camera_help():
    print(
        "\nCould not get a picture from a camera. Things to check:\n"
        "  1. Close other apps that may be using it (Zoom, Teams, the Camera\n"
        "     app, browser tabs with camera access).\n"
        "  2. Windows: Settings > Privacy & security > Camera, and turn on\n"
        "     'Let desktop apps access your camera'.\n"
        "     Mac: System Settings > Privacy & Security > Camera, and allow\n"
        "     Terminal / VS Code.\n"
        "  3. Unplug and re-plug a USB webcam (try another USB port).\n"
        "  4. Run: python pancake_detection.py --list-cameras\n"
        "  5. Make sure you installed 'opencv-python' (not only\n"
        "     'opencv-python-headless', which cannot open windows).\n"
    )


def open_source(source):
    """
    Returns (cap, is_camera, description), or (None, None, None) on failure.
      source == "auto"   -> use the first camera that works
      source is a number -> use that camera index
      anything else      -> treat it as a video file path
    """
    if source == "auto":
        for index in range(MAX_CAMERA_INDEX + 1):
            cap, backend_name = try_open_camera(index)
            if cap is not None:
                return cap, True, f"camera {index} via {backend_name}"
        return None, None, None

    if source.isdigit():
        index = int(source)
        cap, backend_name = try_open_camera(index)
        if cap is not None:
            return cap, True, f"camera {index} via {backend_name}"
        return None, None, None

    if not os.path.exists(source):
        print(f"Video file not found: {source}")
        return None, None, None
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None, None, None
    return cap, False, f"video file {source}"


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

def nothing(_value):
    """Trackbars require a callback, but we read their values ourselves."""
    pass


def create_controls():
    cv2.namedWindow(WINDOW_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_CTRL, 420, 380)
    for name, (start, maxval) in TRACKBARS.items():
        cv2.createTrackbar(name, WINDOW_CTRL, start, maxval, nothing)


def read_settings():
    """Read the current slider values."""
    return {name: cv2.getTrackbarPos(name, WINDOW_CTRL) for name in TRACKBARS}


def make_mask(frame, settings):
    """Colour threshold + clean-up. White pixels = 'pancake-coloured'."""
    lower = np.array([settings["H low"], settings["S low"], settings["V low"]])
    upper = np.array([settings["H high"], settings["S high"], settings["V high"]])

    # Blur a little to reduce camera noise, then convert to HSV
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    # Opening removes specks and cuts thin bridges that glue the pancake to
    # similarly coloured background; closing then fills small holes in it.
    k = max(1, settings["Cleanup px"])
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    return mask


def score_contour(contour):
    """
    Score how pancake-like (round/elliptical) a contour is.

    Returns a dict with the individual measures and the combined "score"
    (0..1), or None if the contour can't be measured.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if area <= 0 or perimeter <= 0 or len(contour) < 5:
        return None

    # 1) Circularity: 4*pi*area / perimeter^2 (1.0 = perfect circle)
    circularity = min((4.0 * math.pi * area / (perimeter ** 2)) / CIRCULARITY_REF, 1.0)

    # 2) Solidity: area / convex hull area (1.0 = no dents or spikes)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = min(area / hull_area, 1.0) if hull_area > 0 else 0.0

    # 3) Ellipse fit: how well the best-fit ellipse matches the blob's area
    try:
        _center, (d1, d2), _angle = cv2.fitEllipse(contour)
    except cv2.error:
        return None
    major, minor = max(d1, d2), min(d1, d2)
    if major <= 0:
        return None
    ellipse_area = math.pi * (major / 2.0) * (minor / 2.0)
    ellipse_fit = min(area, ellipse_area) / max(area, ellipse_area)
    aspect = minor / major

    score = (circularity * solidity * ellipse_fit) ** (1.0 / 3.0)
    if aspect < MIN_ASPECT:
        score *= aspect / MIN_ASPECT   # too thin/stretched to be a pancake

    return {
        "score": float(score),
        "circularity": float(circularity),
        "solidity": float(solidity),
        "ellipse_fit": float(ellipse_fit),
        "aspect": float(aspect),
    }


def trim_spurs(contour, box, image_shape):
    """
    Cut thin strips/spurs off a blob (e.g. a sliver of similarly coloured
    background touching the pancake). The blob is opened with a kernel that
    scales with its own size, so a solid disc survives but narrow attachments
    don't. Returns the trimmed contour, or the original if trimming would
    remove most of the blob.
    """
    x, y, w, h = box
    k = int(0.25 * min(w, h))
    if k < 5:
        return contour
    k |= 1  # kernel size must be odd

    x0, y0 = max(x - k, 0), max(y - k, 0)
    x1, y1 = min(x + w + k, image_shape[1]), min(y + h + k, image_shape[0])
    blob = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.drawContours(blob, [contour], -1, 255, cv2.FILLED, offset=(-x0, -y0))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(blob, cv2.MORPH_OPEN, kernel)
    found, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not found:
        return contour

    trimmed = max(found, key=cv2.contourArea)
    if cv2.contourArea(trimmed) < 0.6 * cv2.contourArea(contour):
        return contour
    return (trimmed + np.array([[[x0, y0]]])).astype(np.int32)


class PancakeDetector:
    """
    Finds the pancake in each frame: colour mask -> candidate blobs -> shape
    score -> pick the best. Remembers the last position so the choice stays
    stable from frame to frame.
    """

    def __init__(self):
        self.last = None    # (cx, cy, area) of the previous detection
        self.lost = 0

    def reset(self):
        self.last = None
        self.lost = 0

    def _continuity(self, cand, diag):
        """0..1: how close this blob is (position + size) to the last pancake."""
        if self.last is None:
            return 1.0
        last_x, last_y, last_area = self.last
        distance = math.hypot(cand["x"] - last_x, cand["y"] - last_y)
        proximity = math.exp(-distance / (0.3 * diag))
        size_ratio = min(cand["area"], last_area) / max(cand["area"], last_area)
        return 0.5 * proximity + 0.5 * size_ratio

    def detect(self, frame, settings):
        """
        Returns (detection, candidates, mask).

        detection  : dict for the chosen pancake, or None
                     {"x", "y", "width", "height", "area", "score", ...}
        candidates : every blob that was scored, best first (detection is
                     one of them). Used to draw the rejected ones.
        mask       : the cleaned colour mask
        """
        mask = make_mask(frame, settings)
        height, width = mask.shape
        frame_area = height * width
        diag = math.hypot(width, height)
        min_score = settings["Min shape %"] / 100.0

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < settings["Min area"] or area > MAX_AREA_FRACTION * frame_area:
                continue

            # Remove thin attachments, then re-check the size
            contour = trim_spurs(contour, cv2.boundingRect(contour), mask.shape)
            area = cv2.contourArea(contour)
            if area < settings["Min area"]:
                continue

            info = score_contour(contour)
            if info is None:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            moments = cv2.moments(contour)
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2

            # A blob cut off by the frame edge has a truncated shape, and big
            # background regions usually look like that. Penalise it.
            if x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1:
                info["score"] *= BORDER_PENALTY

            info.update({
                "x": cx, "y": cy, "width": w, "height": h,
                "box": (x, y, w, h), "area": area, "contour": contour,
                "chosen": False,
            })
            candidates.append(info)

        if not candidates:
            self._miss()
            return None, [], mask

        # Rank: shape quality first, then a mild preference for larger blobs
        # and for blobs near where the pancake was in the last frame.
        largest = max(c["area"] for c in candidates)
        for c in candidates:
            size_term = 0.75 + 0.25 * (c["area"] / largest)
            continuity_term = 0.85 + 0.15 * self._continuity(c, diag)
            c["rank"] = c["score"] * size_term * continuity_term
        candidates.sort(key=lambda c: c["rank"], reverse=True)

        best = next((c for c in candidates if c["score"] >= min_score), None)
        if best is None:
            self._miss()
            return None, candidates, mask

        best["chosen"] = True
        self.last = (best["x"], best["y"], best["area"])
        self.lost = 0
        return best, candidates, mask

    def _miss(self):
        self.lost += 1
        if self.lost >= LOST_FRAMES_RESET:
            self.last = None


def estimate_hsv_range(hsv_frame, x, y, radius=8):
    """
    Work out an HSV range from a small patch around (x, y).

    Hue is kept tight (that's what separates yellow from skin/wood/olive).
    Saturation and value only get a lower bound, so highlights and shading on
    the pancake are still included while duller background is not.
    """
    patch = hsv_frame[max(y - radius, 0): y + radius + 1,
                      max(x - radius, 0): x + radius + 1].reshape(-1, 3)
    if len(patch) == 0:
        return None

    h_low, h_high = np.percentile(patch[:, 0], [2, 98])
    s_low = np.percentile(patch[:, 1], 5)
    v_low = np.percentile(patch[:, 2], 5)

    return {
        "H low": int(max(h_low - 6, 0)),
        "H high": int(min(h_high + 6, 179)),
        "S low": int(max(s_low - 30, 0)),
        "S high": 255,
        "V low": int(max(v_low - 40, 0)),
        "V high": 255,
    }


def set_range_from_click(hsv_frame, x, y):
    """Calibrate the colour sliders from the pixels around the click."""
    ranges = estimate_hsv_range(hsv_frame, x, y)
    if ranges is None:
        return
    for name, value in ranges.items():
        cv2.setTrackbarPos(name, WINDOW_CTRL, value)
    print(f"Calibrated from ({x}, {y}): {ranges}")


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

def draw_text(frame, text, origin, scale=0.55, color=(255, 255, 255)):
    """Text with a dark outline so it's readable on any background."""
    cv2.putText(frame, text, origin, FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, FONT, scale, color, 1, cv2.LINE_AA)


def draw_candidates(frame, candidates):
    """Thin red outlines + match % for blobs that were considered but not chosen."""
    shown = [c for c in candidates if not c["chosen"]][:MAX_DRAWN_CANDIDATES]
    for c in shown:
        cv2.drawContours(frame, [c["contour"]], -1, REJECT_COLOR, 1)
        label = f"{c['score'] * 100:.0f}%"
        draw_text(frame, label, (max(c["x"] - 14, 0), c["y"]), 0.45, REJECT_COLOR)


def draw_detection(frame, detection):
    """Highlight the pancake: tinted fill, outline, box, center, and label."""
    if detection is None:
        return

    x, y, w, h = detection["box"]
    cx, cy = detection["x"], detection["y"]
    contour = detection["contour"]

    # Translucent coloured fill over the pancake
    overlay = frame.copy()
    cv2.drawContours(overlay, [contour], -1, HIGHLIGHT_COLOR, thickness=cv2.FILLED)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, dst=frame)
    cv2.drawContours(frame, [contour], -1, HIGHLIGHT_COLOR, 2)

    # Bounding box, center marker
    cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, 3)
    cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # Label with position, size and match score, on a solid background tab
    label = (f"PANCAKE  center=({cx}, {cy})  size={w}x{h}px  "
             f"match {detection['score'] * 100:.0f}%")
    (text_w, text_h), baseline = cv2.getTextSize(label, FONT, 0.5, 2)
    tab_h = text_h + baseline + 10
    tab_w = text_w + 12
    frame_h, frame_w = frame.shape[:2]

    tab_x = int(min(max(x, 0), max(frame_w - tab_w, 0)))
    tab_y = y - tab_h
    if tab_y < 36:                      # no room above (HUD bar): put it below
        tab_y = min(y + h, frame_h - tab_h - 24)
    cv2.rectangle(frame, (tab_x, tab_y), (tab_x + tab_w, tab_y + tab_h),
                  BOX_COLOR, cv2.FILLED)
    cv2.putText(frame, label, (tab_x + 6, tab_y + text_h + 4),
                FONT, 0.5, (0, 0, 0), 2, cv2.LINE_AA)


def draw_hud(frame, detection, fps, paused):
    """Status bar on top, key hints on the bottom."""
    height, width = frame.shape[:2]

    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (width, 34), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(bar, 0.55, frame, 0.45, 0, dst=frame)

    if detection is not None:
        status, color = "PANCAKE FOUND", (0, 255, 0)
    else:
        status, color = "SEARCHING...", (0, 165, 255)
    cv2.putText(frame, status, (10, 24), FONT, 0.65, color, 2, cv2.LINE_AA)

    right_text = f"{fps:.0f} FPS" + ("  PAUSED" if paused else "")
    (text_w, _), _ = cv2.getTextSize(right_text, FONT, 0.55, 1)
    draw_text(frame, right_text, (width - text_w - 10, 24), 0.55)

    hints = "click: set colour | SPACE pause | M mask | C rejects | S save | Q quit"
    draw_text(frame, hints, (8, height - 10), 0.42)


def render_composite(frame, mask, detection, candidates=(), fps=0.0,
                     paused=False, show_mask=True, show_candidates=True):
    """Build the image shown in the window: camera view (+ mask panel)."""
    view = frame.copy()
    if show_candidates:
        draw_candidates(view, candidates)
    draw_detection(view, detection)
    draw_hud(view, detection, fps, paused)

    if not show_mask:
        return view

    mask_view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    draw_text(mask_view, "MASK (white = pancake colour)", (10, 24), 0.6)
    return np.hstack([view, mask_view])


def resize_for_display(frame):
    height, width = frame.shape[:2]
    if width <= PANEL_WIDTH:
        return frame
    scale = PANEL_WIDTH / width
    return cv2.resize(frame, (PANEL_WIDTH, int(height * scale)))


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Live pancake detector")
    parser.add_argument("--source", default="auto",
                        help="'auto' (default), a camera number like 0 or 1, "
                             "or a path to a video file")
    parser.add_argument("--list-cameras", action="store_true",
                        help="show which cameras work, then exit")
    args = parser.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    print("Opening video source...")
    cap, is_camera, description = open_source(args.source)
    if cap is None:
        print(f"Could not open source: {args.source}")
        if args.source == "auto" or args.source.isdigit():
            print_camera_help()
        return
    print(f"Using {description}")
    print("Tip: click on the middle of the pancake to calibrate the colour.")

    create_controls()
    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_AUTOSIZE)

    detector = PancakeDetector()

    # Shared state so the mouse callback can see the latest HSV frame
    state = {"hsv": None, "width": 0, "height": 0}

    def on_mouse(event, x, y, _flags, _param):
        # Only clicks on the camera panel (left side) count
        if (event == cv2.EVENT_LBUTTONDOWN and state["hsv"] is not None
                and x < state["width"] and y < state["height"]):
            set_range_from_click(state["hsv"], x, y)
            detector.reset()

    cv2.setMouseCallback(WINDOW_MAIN, on_mouse)

    paused = False
    show_mask = True
    show_candidates = True
    frame = None
    detection = None
    candidates = []
    fps = 0.0
    last_time = time.perf_counter()
    failed_reads = 0

    while True:
        if not paused or frame is None:
            ret, new_frame = cap.read()
            if not ret or new_frame is None:
                failed_reads += 1
                if is_camera:
                    if failed_reads > 30:
                        print("Lost the camera feed.")
                        break
                    time.sleep(0.03)
                    continue
                # Video file ended: loop back to the start for repeated tuning
                if failed_reads > 1:
                    print("Could not read any frames from the video.")
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            failed_reads = 0
            frame = resize_for_display(new_frame)

        settings = read_settings()
        state["hsv"] = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        state["height"], state["width"] = frame.shape[:2]

        detection, candidates, mask = detector.detect(frame, settings)

        now = time.perf_counter()
        dt = now - last_time
        last_time = now
        if dt > 0:
            fps = 1.0 / dt if fps == 0 else 0.9 * fps + 0.1 / dt

        cv2.imshow(WINDOW_MAIN, render_composite(
            frame, mask, detection, candidates, fps, paused,
            show_mask, show_candidates))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("m"):
            show_mask = not show_mask
        elif key == ord("c"):
            show_candidates = not show_candidates
        elif key == ord("s"):
            os.makedirs("screenshots", exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join("screenshots", f"pancake_{stamp}.png")
            cv2.imwrite(path, render_composite(
                frame, mask, detection, candidates, fps, paused,
                show_mask, show_candidates))
            print(f"Saved screenshot: {path}")
        elif key == ord("p"):
            if detection:
                print({k: (round(v, 3) if isinstance(v, float) else v)
                       for k, v in detection.items()
                       if k not in ("box", "contour", "chosen", "rank")})
            else:
                print("No pancake detected")
            print(json.dumps(settings))

        # Stop if the user closes the window with the X button
        if cv2.getWindowProperty(WINDOW_MAIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    print("\nFinal HSV settings (save these!):")
    print(json.dumps(read_settings(), indent=2))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()