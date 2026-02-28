#!/usr/bin/env python3
"""Rod Detection GUI App

Connects to a RealSense D435i depth camera, segments non-flat objects
from the depth image, and fits oriented bounding rectangles (x, y, angle)
to each detected object. Includes a demo/simulation mode when no camera
is connected.

Usage:
    python3 rod_detection_app.py [--demo]
"""

import sys
import os
import time
import threading
import argparse
import math
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk

# Add src to path for existing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    rs = None
    REALSENSE_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# Detection logic (standalone, no camera dependency)
# ──────────────────────────────────────────────────────────────────────────────

class DetectedObject:
    """A non-flat object found in the depth image."""

    def __init__(self, cx: float, cy: float, angle_deg: float,
                 width_px: float, height_px: float,
                 area_px: float, contour: np.ndarray,
                 depth_mm: float, aspect_ratio: float):
        self.cx = cx                    # image x center (pixels)
        self.cy = cy                    # image y center (pixels)
        self.angle_deg = angle_deg      # orientation of long axis (degrees, 0=horizontal)
        self.width_px = width_px        # length of long axis (pixels)
        self.height_px = height_px      # length of short axis (pixels)
        self.area_px = area_px          # contour area in pixels
        self.contour = contour          # OpenCV contour
        self.depth_mm = depth_mm        # estimated depth of object center (mm)
        self.aspect_ratio = aspect_ratio


class DepthSegmenter:
    """Segments non-flat objects from aligned color+depth frames."""

    def __init__(self):
        # Depth workspace
        self.depth_min_mm = 200
        self.depth_max_mm = 1200
        # Objects must be this many mm above table to count
        self.table_tolerance_mm = 15
        # Contour size filters
        self.min_area_px = 300
        self.max_area_px = 200000
        # Morphological kernel size
        self.morph_kernel_size = 5

    def _estimate_table_depth(self, depth_image: np.ndarray, workspace_mask: np.ndarray) -> int:
        """Find the dominant depth (table surface) using histogram mode."""
        valid = depth_image[workspace_mask > 0]
        if len(valid) < 100:
            return int((self.depth_min_mm + self.depth_max_mm) / 2)
        # Bin into 5mm buckets for robust mode finding
        hist, edges = np.histogram(valid, bins=range(self.depth_min_mm, self.depth_max_mm, 5))
        if hist.max() == 0:
            return int((self.depth_min_mm + self.depth_max_mm) / 2)
        return int(edges[np.argmax(hist)])

    def segment(self, depth_image: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """Segment objects above the table plane.

        Returns:
            object_mask: binary mask of elevated objects (uint8)
            workspace_mask: binary mask of valid depth region (uint8)
            table_depth_mm: estimated table depth in mm
        """
        # Workspace: valid depth region
        workspace_mask = (
            (depth_image > self.depth_min_mm) &
            (depth_image < self.depth_max_mm)
        ).astype(np.uint8) * 255

        table_depth_mm = self._estimate_table_depth(depth_image, workspace_mask)

        # Objects: pixels significantly closer to camera than table
        object_mask = (
            (depth_image > self.depth_min_mm) &
            (depth_image < table_depth_mm - self.table_tolerance_mm)
        ).astype(np.uint8) * 255

        # Morphological cleanup
        k = self.morph_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)

        return object_mask, workspace_mask, table_depth_mm

    def find_objects(self, depth_image: np.ndarray) -> tuple[list[DetectedObject], np.ndarray, int]:
        """Find all non-flat objects and fit oriented bounding rectangles.

        Returns:
            objects: list of DetectedObject
            object_mask: the segmentation mask used
            table_depth_mm: estimated table depth
        """
        object_mask, _, table_depth_mm = self.segment(depth_image)

        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area_px or area > self.max_area_px:
                continue

            # Fit minimum area bounding rectangle
            rect = cv2.minAreaRect(contour)
            (cx, cy), (w, h), angle = rect

            # Normalize: w >= h, angle in [-90, 90]
            if w < h:
                w, h = h, w
                angle += 90
            # Clamp angle to [-90, 90]
            angle = angle % 180
            if angle > 90:
                angle -= 180

            aspect_ratio = w / h if h > 0 else 1.0

            # Sample depth at center of the object
            cx_int, cy_int = int(np.clip(cx, 0, depth_image.shape[1] - 1)), \
                             int(np.clip(cy, 0, depth_image.shape[0] - 1))
            depth_mm = float(depth_image[cy_int, cx_int])
            if depth_mm == 0:
                # Fallback: median of non-zero depths in contour bounding box
                x0, y0, bw, bh = cv2.boundingRect(contour)
                roi = depth_image[y0:y0+bh, x0:x0+bw]
                nz = roi[roi > 0]
                depth_mm = float(np.median(nz)) if len(nz) > 0 else 0.0

            objects.append(DetectedObject(
                cx=cx, cy=cy,
                angle_deg=angle,
                width_px=w, height_px=h,
                area_px=area,
                contour=contour,
                depth_mm=depth_mm,
                aspect_ratio=aspect_ratio,
            ))

        # Sort by area descending (largest first)
        objects.sort(key=lambda o: o.area_px, reverse=True)
        return objects, object_mask, table_depth_mm


def draw_detections(image: np.ndarray, objects: list[DetectedObject],
                    table_depth_mm: int) -> np.ndarray:
    """Draw detection overlays on the image."""
    vis = image.copy()

    # Draw table depth label
    cv2.putText(vis, f"Table: {table_depth_mm} mm",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

    colors = [
        (0, 255, 0),    # green
        (0, 180, 255),  # orange
        (255, 0, 255),  # magenta
        (0, 255, 255),  # cyan
        (255, 255, 0),  # yellow
    ]

    for i, obj in enumerate(objects):
        color = colors[i % len(colors)]

        # Draw contour
        cv2.drawContours(vis, [obj.contour], -1, color, 2)

        # Draw oriented bounding rectangle
        rect = cv2.minAreaRect(obj.contour)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)
        cv2.drawContours(vis, [box], 0, color, 1)

        # Draw axis line along long dimension
        cx, cy = int(obj.cx), int(obj.cy)
        half_len = int(obj.width_px / 2)
        angle_rad = math.radians(obj.angle_deg)
        dx = int(half_len * math.cos(angle_rad))
        dy = int(half_len * math.sin(angle_rad))
        cv2.line(vis, (cx - dx, cy - dy), (cx + dx, cy + dy), color, 2)

        # Draw center dot
        cv2.circle(vis, (cx, cy), 5, color, -1)

        # Label: object index, angle, depth
        label = f"#{i+1} {obj.angle_deg:.0f}deg {obj.depth_mm:.0f}mm"
        text_y = max(cy - 12, 14)
        cv2.putText(vis, label, (cx + 8, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return vis


def depth_colormap(depth_image: np.ndarray) -> np.ndarray:
    """Convert depth uint16 image to a colored visualization."""
    # Normalize to 0-255 ignoring zeros
    d = depth_image.astype(np.float32)
    mask = d > 0
    if mask.any():
        d_min, d_max = d[mask].min(), d[mask].max()
        if d_max > d_min:
            d = np.where(mask, (d - d_min) / (d_max - d_min) * 255, 0)
        else:
            d = np.where(mask, 128.0, 0.0)
    d8 = d.astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_JET)


# ──────────────────────────────────────────────────────────────────────────────
# Demo mode: synthetic scene generator
# ──────────────────────────────────────────────────────────────────────────────

class DemoSceneGenerator:
    """Generates synthetic color+depth frames with a simulated rod."""

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self._t = 0.0
        self._rng = np.random.default_rng(42)

    def get_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (color_bgr, depth_mm) synthetic frames."""
        self._t += 0.04

        w, h = self.width, self.height
        TABLE_DEPTH = 700  # mm

        # --- depth image: flat table ---
        depth = np.full((h, w), TABLE_DEPTH, dtype=np.uint16)
        # Add some noise
        noise = self._rng.integers(-8, 8, size=(h, w), dtype=np.int16)
        depth = np.clip(depth.astype(np.int32) + noise, 0, 65535).astype(np.uint16)

        # --- color image: gray table surface ---
        color = np.full((h, w, 3), (90, 85, 80), dtype=np.uint8)

        # Subtle texture on table
        tex = self._rng.integers(0, 12, (h, w, 1), dtype=np.uint8)
        color = np.clip(color.astype(np.int16) + tex - 6, 0, 255).astype(np.uint8)

        # --- Animated rod ---
        rod_angle = math.radians(30 + 20 * math.sin(self._t * 0.7))  # oscillate
        rod_cx = int(w * 0.5 + 60 * math.sin(self._t * 0.3))
        rod_cy = int(h * 0.5 + 30 * math.cos(self._t * 0.4))
        rod_length = 180  # pixels
        rod_radius = 12   # pixels (half-width)
        rod_depth = 680   # mm (slightly above table)

        # Build rod mask using rotated rectangle
        cos_a, sin_a = math.cos(rod_angle), math.sin(rod_angle)
        # Four corners of the rod rect
        half_l = rod_length / 2
        corners = [
            (rod_cx + half_l * cos_a - rod_radius * sin_a,
             rod_cy + half_l * sin_a + rod_radius * cos_a),
            (rod_cx + half_l * cos_a + rod_radius * sin_a,
             rod_cy + half_l * sin_a - rod_radius * cos_a),
            (rod_cx - half_l * cos_a + rod_radius * sin_a,
             rod_cy - half_l * sin_a - rod_radius * cos_a),
            (rod_cx - half_l * cos_a - rod_radius * sin_a,
             rod_cy - half_l * sin_a + rod_radius * cos_a),
        ]
        pts = np.array(corners, dtype=np.int32)
        rod_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(rod_mask, [pts], 255)

        # Apply rod to depth (rod is closer to camera)
        depth[rod_mask > 0] = rod_depth

        # Apply rod to color (dark metallic appearance)
        rod_color_base = np.array([30, 35, 40], dtype=np.uint8)  # dark blue-gray
        ys, xs = np.where(rod_mask > 0)
        for i in range(len(ys)):
            # Slight sheen highlight along center
            perp = abs((xs[i] - rod_cx) * (-sin_a) + (ys[i] - rod_cy) * cos_a)
            brightness = max(0, 1.0 - perp / rod_radius)
            c = (rod_color_base * (1.0 + 0.4 * brightness)).clip(0, 255).astype(np.uint8)
            color[ys[i], xs[i]] = c

        # --- Second object: a small box ---
        box_cx, box_cy = int(w * 0.25), int(h * 0.65)
        box_half = 30
        bx0 = max(0, box_cx - box_half)
        bx1 = min(w, box_cx + box_half)
        by0 = max(0, box_cy - box_half)
        by1 = min(h, box_cy + box_half)
        depth[by0:by1, bx0:bx1] = 660
        color[by0:by1, bx0:bx1] = [100, 80, 60]  # brownish box

        return color, depth


# ──────────────────────────────────────────────────────────────────────────────
# Camera thread
# ──────────────────────────────────────────────────────────────────────────────

class CameraThread(threading.Thread):
    """Background thread that captures frames and runs detection."""

    def __init__(self, demo_mode: bool = False):
        super().__init__(daemon=True)
        self.demo_mode = demo_mode
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Outputs updated by thread, read by GUI
        self._color_image: np.ndarray | None = None
        self._depth_image: np.ndarray | None = None
        self._objects: list[DetectedObject] = []
        self._table_depth_mm: int = 0
        self._fps: float = 0.0
        self._error: str | None = None
        self._running: bool = False

        self.segmenter = DepthSegmenter()

        # RealSense objects (real mode)
        self._pipeline = None
        self._align = None

        # Demo scene
        self._demo_gen: DemoSceneGenerator | None = None

    def configure_segmenter(self, **kwargs):
        """Update segmenter parameters thread-safely."""
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self.segmenter, k):
                    setattr(self.segmenter, k, v)

    def get_latest(self):
        """Return (color_bgr, depth_mm, objects, table_depth_mm, fps, error, running)."""
        with self._lock:
            return (
                self._color_image,
                self._depth_image,
                list(self._objects),
                self._table_depth_mm,
                self._fps,
                self._error,
                self._running,
            )

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            if self.demo_mode:
                self._run_demo()
            else:
                self._run_realsense()
        except Exception as e:
            with self._lock:
                self._error = str(e)
                self._running = False

    def _run_demo(self):
        self._demo_gen = DemoSceneGenerator()
        with self._lock:
            self._running = True
        t_prev = time.time()
        while not self._stop_event.is_set():
            color, depth = self._demo_gen.get_frames()
            objects, _, table_depth = self.segmenter.find_objects(depth)
            t_now = time.time()
            fps = 1.0 / max(t_now - t_prev, 0.001)
            t_prev = t_now
            with self._lock:
                self._color_image = color
                self._depth_image = depth
                self._objects = objects
                self._table_depth_mm = table_depth
                self._fps = fps
            time.sleep(0.05)  # ~20 fps demo
        with self._lock:
            self._running = False

    def _run_realsense(self):
        if not REALSENSE_AVAILABLE:
            raise RuntimeError("pyrealsense2 is not installed")

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)

        try:
            pipeline.start(config)
            align = rs.align(rs.stream.color)

            with self._lock:
                self._running = True

            t_prev = time.time()
            while not self._stop_event.is_set():
                frames = pipeline.wait_for_frames(timeout_ms=3000)
                aligned = align.process(frames)
                cf = aligned.get_color_frame()
                df = aligned.get_depth_frame()
                if not cf or not df:
                    continue

                color = np.asanyarray(cf.get_data())
                depth = np.asanyarray(df.get_data())

                objects, _, table_depth = self.segmenter.find_objects(depth)
                t_now = time.time()
                fps = 1.0 / max(t_now - t_prev, 0.001)
                t_prev = t_now

                with self._lock:
                    self._color_image = color
                    self._depth_image = depth
                    self._objects = objects
                    self._table_depth_mm = table_depth
                    self._fps = fps
        finally:
            pipeline.stop()
            with self._lock:
                self._running = False


# ──────────────────────────────────────────────────────────────────────────────
# GUI
# ──────────────────────────────────────────────────────────────────────────────

IMG_W, IMG_H = 320, 240  # Display size for each panel


def cv2_to_tk(img_bgr: np.ndarray, w: int, h: int) -> ImageTk.PhotoImage:
    """Convert a BGR OpenCV image to a Tkinter PhotoImage, resized to (w, h)."""
    resized = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(pil_img)


def mask_to_tk(mask: np.ndarray, w: int, h: int) -> ImageTk.PhotoImage:
    """Convert a single-channel mask to a colored Tkinter image."""
    colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # Tint: white pixels become greenish
    colored[:, :, 0] = 0
    colored[:, :, 2] = 0
    return cv2_to_tk(colored, w, h)


class RodDetectionApp:
    """Main tkinter application window."""

    def __init__(self, root: tk.Tk, demo_mode: bool = False):
        self.root = root
        self.demo_mode = demo_mode
        self.camera_thread: CameraThread | None = None

        root.title("Rod Detection - Depth Camera Segmenter")
        root.resizable(True, True)
        root.configure(bg="#1e1e2e")

        self._build_ui()
        self._update_loop()

    def _build_ui(self):
        """Construct all UI elements."""
        root = self.root
        BG = "#1e1e2e"
        FG = "#cdd6f4"
        PANEL_BG = "#181825"
        ACCENT = "#89b4fa"
        BTN_BG = "#313244"
        BTN_FG = "#cdd6f4"

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background=BG, foreground=FG, font=("Helvetica", 10))
        style.configure("Header.TLabel", background=BG, foreground=ACCENT,
                        font=("Helvetica", 11, "bold"))
        style.configure("Status.TLabel", background=PANEL_BG, foreground=FG,
                        font=("Courier", 10))
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL_BG)
        style.configure("TScale", background=BG, troughcolor="#45475a", slidercolor=ACCENT)
        style.configure("TButton", background=BTN_BG, foreground=BTN_FG,
                        font=("Helvetica", 10, "bold"), borderwidth=0)
        style.map("TButton",
                  background=[("active", "#585b70"), ("pressed", "#1e66f5")],
                  foreground=[("active", FG)])
        style.configure("Treeview", background=PANEL_BG, foreground=FG,
                        fieldbackground=PANEL_BG, rowheight=22,
                        font=("Courier", 9))
        style.configure("Treeview.Heading", background="#313244", foreground=ACCENT,
                        font=("Helvetica", 9, "bold"))
        style.map("Treeview", background=[("selected", "#313244")])

        # ── Top bar ──
        top_bar = ttk.Frame(root)
        top_bar.pack(fill=tk.X, padx=8, pady=(8, 4))

        mode_txt = "DEMO MODE" if self.demo_mode else "LIVE CAMERA"
        mode_color = "#a6e3a1" if self.demo_mode else "#f38ba8"
        self.lbl_mode = tk.Label(top_bar, text=mode_txt, bg=mode_color,
                                  fg="#1e1e2e", font=("Helvetica", 10, "bold"),
                                  padx=6, pady=2)
        self.lbl_mode.pack(side=tk.LEFT, padx=(0, 10))

        self.lbl_fps = tk.Label(top_bar, text="FPS: --", bg=BG, fg=ACCENT,
                                 font=("Courier", 11, "bold"))
        self.lbl_fps.pack(side=tk.LEFT, padx=4)

        self.lbl_status = tk.Label(top_bar, text="Idle", bg=BG, fg=FG,
                                    font=("Helvetica", 10))
        self.lbl_status.pack(side=tk.LEFT, padx=10)

        self.btn_start = ttk.Button(top_bar, text="Start", width=8,
                                     command=self._on_start)
        self.btn_start.pack(side=tk.RIGHT, padx=4)
        self.btn_stop = ttk.Button(top_bar, text="Stop", width=8,
                                    command=self._on_stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.RIGHT, padx=4)
        self.btn_snap = ttk.Button(top_bar, text="Snapshot", width=10,
                                    command=self._on_snapshot, state=tk.DISABLED)
        self.btn_snap.pack(side=tk.RIGHT, padx=4)

        # ── Image panels ──
        img_frame = ttk.Frame(root)
        img_frame.pack(fill=tk.X, padx=8, pady=4)

        def make_img_panel(parent, title: str):
            f = ttk.Frame(parent, style="Panel.TFrame", padding=4)
            f.pack(side=tk.LEFT, padx=4, expand=False)
            ttk.Label(f, text=title, style="Header.TLabel").pack()
            lbl = tk.Label(f, bg=PANEL_BG, width=IMG_W, height=IMG_H)
            lbl.pack()
            # Placeholder
            ph = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
            ph[:] = (40, 40, 60)
            cv2.putText(ph, "No signal", (IMG_W//2 - 40, IMG_H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 160), 1)
            tk_img = cv2_to_tk(ph, IMG_W, IMG_H)
            lbl.configure(image=tk_img)
            lbl._tk_img = tk_img
            return lbl

        self.lbl_color = make_img_panel(img_frame, "RGB + Detections")
        self.lbl_depth = make_img_panel(img_frame, "Depth (colormap)")
        self.lbl_mask  = make_img_panel(img_frame, "Segmentation Mask")

        # ── Bottom: parameters + detection table ──
        bottom = ttk.Frame(root)
        bottom.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Left: parameters
        params_frame = ttk.Frame(bottom, style="Panel.TFrame", padding=8)
        params_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        ttk.Label(params_frame, text="Detection Parameters", style="Header.TLabel").pack(anchor=tk.W)
        ttk.Separator(params_frame).pack(fill=tk.X, pady=4)

        self._param_vars = {}

        def add_slider(parent, label, key, lo, hi, default, dtype=int, resolution=1):
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=22, anchor=tk.W).pack(side=tk.LEFT)
            var = tk.DoubleVar(value=default)
            self._param_vars[key] = (var, dtype)
            val_lbl = tk.Label(row, text=str(default), width=6,
                                bg=PANEL_BG, fg=ACCENT, font=("Courier", 9))
            val_lbl.pack(side=tk.RIGHT)

            def on_change(v, lbl=val_lbl, k=key, dt=dtype):
                val = dt(float(v))
                lbl.configure(text=str(val))
                self._apply_params()

            sl = ttk.Scale(row, from_=lo, to=hi, variable=var,
                           orient=tk.HORIZONTAL, length=160,
                           command=on_change)
            sl.pack(side=tk.LEFT, padx=4)
            return var

        add_slider(params_frame, "Depth min (mm)",    "depth_min_mm",    100, 600,  200)
        add_slider(params_frame, "Depth max (mm)",    "depth_max_mm",    400, 2000, 1200)
        add_slider(params_frame, "Table tolerance (mm)", "table_tolerance_mm", 5, 100, 15)
        add_slider(params_frame, "Min area (px)",     "min_area_px",     100, 5000, 300)
        add_slider(params_frame, "Max area (px)",     "max_area_px",     1000, 200000, 200000)
        add_slider(params_frame, "Morph kernel",      "morph_kernel_size", 1, 15, 5)

        # Right: detections table
        table_frame = ttk.Frame(bottom, style="Panel.TFrame", padding=8)
        table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(table_frame, text="Detected Objects", style="Header.TLabel").pack(anchor=tk.W)
        ttk.Separator(table_frame).pack(fill=tk.X, pady=4)

        cols = ("#", "X (px)", "Y (px)", "Angle (deg)", "W (px)", "H (px)",
                "Aspect", "Area (px)", "Depth (mm)")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings",
                                  height=8)
        col_widths = [30, 70, 70, 90, 70, 70, 60, 80, 80]
        for col, cw in zip(cols, col_widths):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=cw, anchor=tk.CENTER, minwidth=40)
        self.tree.pack(fill=tk.BOTH, expand=True)

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Status bar
        status_bar = tk.Frame(root, bg="#11111b", height=24)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.lbl_bottom = tk.Label(status_bar, text="Ready. Press Start to begin.",
                                    bg="#11111b", fg="#6c7086",
                                    font=("Courier", 9), anchor=tk.W)
        self.lbl_bottom.pack(side=tk.LEFT, padx=8)
        self.lbl_obj_count = tk.Label(status_bar, text="Objects: 0",
                                       bg="#11111b", fg=ACCENT,
                                       font=("Courier", 9))
        self.lbl_obj_count.pack(side=tk.RIGHT, padx=8)

    def _apply_params(self):
        """Push slider values to the segmenter."""
        if self.camera_thread is None:
            return
        kwargs = {k: dt(v.get()) for k, (v, dt) in self._param_vars.items()}
        self.camera_thread.configure_segmenter(**kwargs)

    def _on_start(self):
        """Start camera capture thread."""
        if self.camera_thread and self.camera_thread.is_alive():
            return
        self.camera_thread = CameraThread(demo_mode=self.demo_mode)
        # Apply current slider values
        kwargs = {k: dt(v.get()) for k, (v, dt) in self._param_vars.items()}
        self.camera_thread.segmenter.__dict__.update(kwargs)
        self.camera_thread.start()

        self.btn_start.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        self.btn_snap.configure(state=tk.NORMAL)
        self.lbl_status.configure(text="Running...")

    def _on_stop(self):
        """Stop camera capture thread."""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.join(timeout=2.0)
            self.camera_thread = None
        self.btn_start.configure(state=tk.NORMAL)
        self.btn_stop.configure(state=tk.DISABLED)
        self.btn_snap.configure(state=tk.DISABLED)
        self.lbl_status.configure(text="Stopped")

    def _on_snapshot(self):
        """Save current frame to disk."""
        if self.camera_thread is None:
            return
        color, depth, objects, table_depth, fps, error, running = \
            self.camera_thread.get_latest()
        if color is None:
            messagebox.showinfo("Snapshot", "No frame available yet.")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(os.path.dirname(__file__), f"snapshot_{ts}.png")
        vis = draw_detections(color, objects, table_depth)
        cv2.imwrite(path, vis)
        self.lbl_bottom.configure(text=f"Saved snapshot: {path}")
        messagebox.showinfo("Snapshot", f"Saved:\n{path}")

    def _update_loop(self):
        """Periodic GUI update called from main thread."""
        try:
            self._refresh_display()
        except Exception:
            pass
        self.root.after(50, self._update_loop)  # ~20 Hz GUI refresh

    def _refresh_display(self):
        """Pull latest data from camera thread and update all widgets."""
        if self.camera_thread is None:
            return

        color, depth, objects, table_depth, fps, error, running = \
            self.camera_thread.get_latest()

        # Handle errors
        if error:
            self.lbl_status.configure(text=f"Error: {error}")
            self._on_stop()
            return

        if running:
            self.lbl_status.configure(text="Running...")
        elif not running and self.camera_thread is not None:
            # Thread has fully exited
            if not self.camera_thread.is_alive():
                self.lbl_status.configure(text="Stopped")
                self.btn_start.configure(state=tk.NORMAL)
                self.btn_stop.configure(state=tk.DISABLED)
                self.btn_snap.configure(state=tk.DISABLED)

        self.lbl_fps.configure(text=f"FPS: {fps:.1f}")

        if color is None or depth is None:
            return

        # --- Color + detections ---
        vis = draw_detections(color, objects, table_depth)
        tk_color = cv2_to_tk(vis, IMG_W, IMG_H)
        self.lbl_color.configure(image=tk_color)
        self.lbl_color._tk_img = tk_color

        # --- Depth colormap ---
        dcmap = depth_colormap(depth)
        # Overlay table line indicator
        if table_depth > 0:
            cv2.putText(dcmap, f"Table:{table_depth}mm",
                        (4, dcmap.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tk_depth = cv2_to_tk(dcmap, IMG_W, IMG_H)
        self.lbl_depth.configure(image=tk_depth)
        self.lbl_depth._tk_img = tk_depth

        # --- Segmentation mask ---
        seg = self.camera_thread.segmenter
        obj_mask, _, _ = seg.segment(depth)
        # Colorize: green for objects
        mask_vis = np.zeros((*obj_mask.shape, 3), dtype=np.uint8)
        mask_vis[obj_mask > 0] = (0, 220, 80)
        # Show contours
        contours = [o.contour for o in objects]
        cv2.drawContours(mask_vis, contours, -1, (255, 255, 255), 1)
        tk_mask = cv2_to_tk(mask_vis, IMG_W, IMG_H)
        self.lbl_mask.configure(image=tk_mask)
        self.lbl_mask._tk_img = tk_mask

        # --- Detection table ---
        for row in self.tree.get_children():
            self.tree.delete(row)
        for i, obj in enumerate(objects):
            tag = "even" if i % 2 == 0 else "odd"
            self.tree.insert("", tk.END, values=(
                f"{i+1}",
                f"{obj.cx:.1f}",
                f"{obj.cy:.1f}",
                f"{obj.angle_deg:.1f}°",
                f"{obj.width_px:.1f}",
                f"{obj.height_px:.1f}",
                f"{obj.aspect_ratio:.2f}",
                f"{obj.area_px:.0f}",
                f"{obj.depth_mm:.0f}",
            ), tags=(tag,))
        self.tree.tag_configure("even", background="#1e1e2e")
        self.tree.tag_configure("odd",  background="#181825")

        # --- Status bar ---
        self.lbl_obj_count.configure(text=f"Objects: {len(objects)}")
        if objects:
            best = objects[0]
            self.lbl_bottom.configure(
                text=f"Largest: x={best.cx:.0f} y={best.cy:.0f} "
                     f"angle={best.angle_deg:.1f}deg "
                     f"depth={best.depth_mm:.0f}mm "
                     f"AR={best.aspect_ratio:.1f}")
        else:
            self.lbl_bottom.configure(text="No objects detected above table")

    def on_close(self):
        """Clean up on window close."""
        if self.camera_thread:
            self.camera_thread.stop()
        self.root.destroy()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rod Detection GUI App")
    parser.add_argument("--demo", action="store_true",
                        help="Run in demo mode (no camera required)")
    args = parser.parse_args()

    demo_mode = args.demo
    if not demo_mode and not REALSENSE_AVAILABLE:
        print("WARNING: pyrealsense2 not available. Falling back to demo mode.")
        demo_mode = True

    root = tk.Tk()
    app = RodDetectionApp(root, demo_mode=demo_mode)
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    mode_str = "DEMO" if demo_mode else "LIVE"
    print(f"[RodDetectionApp] Starting in {mode_str} mode")
    print("  Press 'Start' to begin capture.")
    print("  Press 'Stop' to pause.")
    print("  Press 'Snapshot' to save a PNG of the current detection frame.")
    print("  Close window to exit.")

    # Auto-start in demo mode for convenience
    if demo_mode:
        root.after(400, app._on_start)

    root.mainloop()


if __name__ == "__main__":
    main()
