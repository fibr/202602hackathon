"""Calibration board detection for intrinsics and extrinsics calibration.

Supports:
  - CharucoBoard: Combined checkerboard + ArUco markers (robust, partial occlusion OK)
  - Checkerboard: Traditional black/white checkerboard (legacy fallback)

The BoardDetector class provides a unified interface regardless of board type.

Usage:
    # From config
    detector = BoardDetector.from_config(config)

    # Detect corners in a grayscale image
    result = detector.detect(gray)
    if result is not None:
        corners, ids = result.corners, result.ids

    # Compute board pose (PnP)
    T, obj_pts, err = detector.compute_pose(result, intrinsics)

    # Calibrate intrinsics from multiple detections
    ret, intr = detector.calibrate_intrinsics(detections, image_size)
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from cv2 import aruco


@dataclass
class BoardDetection:
    """Result of a board detection in a single image.

    Attributes:
        corners: Nx1x2 array of detected corner positions in pixels.
        ids: Nx1 array of corner IDs (charuco) or None (checkerboard).
        board_size: (cols, rows) of detected inner corners.
        is_partial: True if not all corners were found (charuco only).
    """
    corners: np.ndarray
    ids: Optional[np.ndarray]
    board_size: tuple
    is_partial: bool = False


class BoardDetector:
    """Unified calibration board detector.

    Wraps both CharucoBoard and traditional checkerboard detection behind
    a common interface so the calibration pipeline doesn't need to know
    which board type is in use.

    Args:
        board_type: 'charuco' or 'checkerboard'
        cols: Number of squares along X (charuco) or inner corners (checkerboard)
        rows: Number of squares along Y (charuco) or inner corners (checkerboard)
        square_size_m: Square side length in meters
        marker_size_m: ArUco marker side length in meters (charuco only)
        dictionary_name: ArUco dictionary name, e.g. 'DICT_4X4_250' (charuco only)
    """

    def __init__(self, board_type: str = 'charuco',
                 cols: int = 13, rows: int = 9,
                 square_size_m: float = 0.020,
                 marker_size_m: float = 0.015,
                 dictionary_name: str = 'DICT_4X4_250',
                 legacy_pattern: bool = False):
        self.board_type = board_type.lower()
        self.square_size_m = square_size_m
        self.marker_size_m = marker_size_m

        if self.board_type == 'charuco':
            # cols x rows = number of squares; inner corners = (cols-1) x (rows-1)
            self.board_cols = cols       # squares
            self.board_rows = rows       # squares
            self.inner_cols = cols - 1   # inner corners
            self.inner_rows = rows - 1   # inner corners

            dict_id = getattr(aruco, dictionary_name, None)
            if dict_id is None:
                raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
            self._dictionary = aruco.getPredefinedDictionary(dict_id)
            self._charuco_board = aruco.CharucoBoard(
                (cols, rows), square_size_m, marker_size_m, self._dictionary)
            # Legacy pattern for boards printed before OpenCV 4.7 (incl. calib.io)
            self._legacy_pattern = legacy_pattern
            if legacy_pattern:
                self._charuco_board.setLegacyPattern(True)
            self._detector = aruco.CharucoDetector(self._charuco_board)
            self._init_detector_params()
        elif self.board_type == 'checkerboard':
            # For checkerboard, cols/rows ARE the inner corners directly
            self.inner_cols = cols
            self.inner_rows = rows
            self.board_cols = cols + 1   # squares
            self.board_rows = rows + 1   # squares
            self._charuco_board = None
            self._detector = None
        else:
            raise ValueError(f"Unknown board type: {board_type!r} "
                             "(expected 'charuco' or 'checkerboard')")

    def _init_detector_params(self):
        """Configure ArUco detector parameters for real-world conditions."""
        det_params = self._detector.getDetectorParameters()
        det_params.adaptiveThreshWinSizeMin = 3
        det_params.adaptiveThreshWinSizeMax = 23
        det_params.adaptiveThreshWinSizeStep = 10
        det_params.adaptiveThreshConstant = 7
        self._detector.setDetectorParameters(det_params)

    def _rebuild_detector(self):
        """Rebuild CharucoDetector after changing board settings."""
        self._detector = aruco.CharucoDetector(self._charuco_board)
        self._init_detector_params()

    @classmethod
    def from_config(cls, config: dict) -> 'BoardDetector':
        """Create a BoardDetector from the project config dict.

        Reads the 'calibration_board' section. Falls back to a 7x9
        checkerboard with 20mm squares if no config is found.

        Args:
            config: Full project config (e.g. from load_config())

        Returns:
            Configured BoardDetector instance
        """
        bc = config.get('calibration_board', {})
        board_type = bc.get('type', 'checkerboard')

        if board_type == 'charuco':
            return cls(
                board_type='charuco',
                cols=bc.get('cols', 13),
                rows=bc.get('rows', 9),
                square_size_m=bc.get('square_size_mm', 20.0) / 1000.0,
                marker_size_m=bc.get('marker_size_mm', 15.0) / 1000.0,
                dictionary_name=bc.get('dictionary', 'DICT_4X4_250'),
                legacy_pattern=bc.get('legacy_pattern', False),
            )
        else:
            # Legacy checkerboard: cols/rows are inner corners
            return cls(
                board_type='checkerboard',
                cols=bc.get('cols', 7),
                rows=bc.get('rows', 9),
                square_size_m=bc.get('square_size_mm', 20.0) / 1000.0,
            )

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, gray: np.ndarray) -> Optional[BoardDetection]:
        """Detect calibration board corners in a grayscale image.

        Args:
            gray: Grayscale image (H, W) uint8

        Returns:
            BoardDetection if corners found, None otherwise.
        """
        if self.board_type == 'charuco':
            return self._detect_charuco(gray)
        else:
            return self._detect_checkerboard(gray)

    def _detect_charuco(self, gray: np.ndarray) -> Optional[BoardDetection]:
        """Detect CharucoBoard corners using the CharucoDetector API.

        Automatically tries legacy marker pattern if the default fails,
        since many printed boards use the pre-OpenCV-4.7 layout.
        """
        charuco_corners, charuco_ids, marker_corners, marker_ids = (
            self._detector.detectBoard(gray))

        n_markers = len(marker_corners) if marker_corners is not None else 0
        n_charuco = len(charuco_corners) if charuco_corners is not None else 0

        # Auto-detect legacy pattern if detection fails
        if n_charuco < 4 and not self._legacy_pattern:
            self._charuco_board.setLegacyPattern(True)
            self._legacy_pattern = True
            self._rebuild_detector()
            reason = (f"{n_markers} markers but {n_charuco} corners"
                      if n_markers > 0 else "no markers found")
            print(f"  CharucoDetector: {reason} — trying legacy pattern")
            charuco_corners, charuco_ids, marker_corners, marker_ids = (
                self._detector.detectBoard(gray))
            n_markers = len(marker_corners) if marker_corners is not None else 0
            n_charuco = len(charuco_corners) if charuco_corners is not None else 0
            if n_charuco < 4 and n_markers == 0:
                # Legacy didn't help either — switch back
                self._charuco_board.setLegacyPattern(False)
                self._legacy_pattern = False
                self._rebuild_detector()

        if n_charuco < 4:
            return None

        is_partial = n_charuco < self.inner_cols * self.inner_rows
        return BoardDetection(
            corners=charuco_corners.reshape(-1, 1, 2).astype(np.float32),
            ids=charuco_ids.flatten() if charuco_ids is not None else None,
            board_size=(self.inner_cols, self.inner_rows),
            is_partial=is_partial,
        )

    def _detect_checkerboard(self, gray: np.ndarray) -> Optional[BoardDetection]:
        """Detect traditional checkerboard using multiple fallback strategies."""
        size = (self.inner_cols, self.inner_rows)

        # Strategy 1: SB detector (most accurate)
        try:
            found, corners = cv2.findChessboardCornersSB(
                gray, size,
                cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
            if found:
                return BoardDetection(corners=corners, ids=None,
                                      board_size=size)
        except cv2.error:
            pass

        # Strategy 2: CLAHE + adaptive threshold
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(enhanced, size, flags)
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                       criteria)
            return BoardDetection(corners=corners, ids=None,
                                  board_size=size)

        # Strategy 3: Sharpen + filter quads
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                 cv2.CALIB_CB_NORMALIZE_IMAGE |
                 cv2.CALIB_CB_FILTER_QUADS)
        found, corners = cv2.findChessboardCorners(sharpened, size, flags)
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                       criteria)
            return BoardDetection(corners=corners, ids=None,
                                  board_size=size)

        # Strategy 4: Reduced-size subsets
        for dc, dr in [(0, -2), (-2, 0), (-2, -2)]:
            sub_size = (self.inner_cols + dc, self.inner_rows + dr)
            if sub_size[0] < 3 or sub_size[1] < 3:
                continue
            try:
                found, corners = cv2.findChessboardCornersSB(
                    gray, sub_size, cv2.CALIB_CB_EXHAUSTIVE)
                if found:
                    return BoardDetection(
                        corners=corners, ids=None,
                        board_size=sub_size, is_partial=True)
            except cv2.error:
                pass

        return None

    # ------------------------------------------------------------------
    # Object points generation
    # ------------------------------------------------------------------

    def get_object_points(self, detection: BoardDetection) -> np.ndarray:
        """Get 3D object points for a detection.

        For charuco boards, uses the corner IDs to look up exact positions.
        For checkerboards, generates a regular grid at the detected size.

        Args:
            detection: A BoardDetection from detect()

        Returns:
            Nx3 float32 array of 3D points (z=0 plane, in meters)
        """
        if self.board_type == 'charuco' and detection.ids is not None:
            # Use CharucoBoard's built-in object point lookup
            all_obj_pts = self._charuco_board.getChessboardCorners()
            obj_pts = np.array([all_obj_pts[i] for i in detection.ids],
                               dtype=np.float32)
            return obj_pts
        else:
            # Regular grid for checkerboard
            cols, rows = detection.board_size
            obj_pts = np.zeros((rows * cols, 3), dtype=np.float32)
            for r in range(rows):
                for c in range(cols):
                    obj_pts[r * cols + c] = [
                        c * self.square_size_m,
                        r * self.square_size_m,
                        0]
            return obj_pts

    # ------------------------------------------------------------------
    # Pose estimation
    # ------------------------------------------------------------------

    def compute_pose(self, detection: BoardDetection, intrinsics):
        """Compute the board-to-camera transform via solvePnP.

        Args:
            detection: A BoardDetection from detect()
            intrinsics: CameraIntrinsics (needs .fx, .fy, .ppx, .ppy, .coeffs)

        Returns:
            (T_board_in_cam, obj_points, reproj_error_px) or (None, None, None)
        """
        obj_pts = self.get_object_points(detection)
        corners_2d = detection.corners.reshape(-1, 2)

        if len(obj_pts) < 4:
            return None, None, None

        camera_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, corners_2d.astype(np.float64),
            camera_matrix, dist_coeffs)
        if not ok:
            return None, None, None

        # Reprojection error
        projected, _ = cv2.projectPoints(
            obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
        reproj_err = float(np.sqrt(np.mean(
            (corners_2d - projected.reshape(-1, 2)) ** 2)))

        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        return T, obj_pts, reproj_err

    # ------------------------------------------------------------------
    # Intrinsics calibration
    # ------------------------------------------------------------------

    def calibrate_intrinsics(self, detections: list, image_size: tuple):
        """Calibrate camera intrinsics from multiple board detections.

        For charuco boards, uses cv2.calibrateCamera with per-frame
        object points looked up by corner ID. For checkerboards, uses the
        same function with a regular grid.

        Args:
            detections: List of BoardDetection objects (one per captured frame)
            image_size: (width, height) of the images

        Returns:
            (rms_error, CameraIntrinsics) tuple.
            The CameraIntrinsics import is deferred to avoid circular deps.

        Raises:
            ValueError: If fewer than 5 detections are provided.
        """
        if len(detections) < 5:
            raise ValueError(
                f"Need at least 5 frames for intrinsics (have {len(detections)})")

        obj_points_list = []
        img_points_list = []
        for det in detections:
            obj_pts = self.get_object_points(det)
            img_pts = det.corners.reshape(-1, 1, 2).astype(np.float32)
            obj_points_list.append(obj_pts)
            img_points_list.append(img_pts)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points_list, img_points_list, image_size, None, None)

        # Build CameraIntrinsics (import here to avoid circular dependency)
        from vision.camera import CameraIntrinsics
        calib = CameraIntrinsics(
            fx=mtx[0, 0], fy=mtx[1, 1],
            ppx=mtx[0, 2], ppy=mtx[1, 2],
            coeffs=dist.ravel().tolist())
        calib.width = image_size[0]
        calib.height = image_size[1]
        return ret, calib

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw_corners(self, image: np.ndarray,
                     detection: BoardDetection) -> np.ndarray:
        """Draw detected corners on the image.

        Args:
            image: BGR image to draw on (modified in place)
            detection: Detection result

        Returns:
            The image with drawn corners
        """
        if self.board_type == 'charuco' and detection.ids is not None:
            aruco.drawDetectedCornersCharuco(
                image, detection.corners, detection.ids)
        else:
            cv2.drawChessboardCorners(
                image, detection.board_size, detection.corners, True)
        return image

    # ------------------------------------------------------------------
    # Description
    # ------------------------------------------------------------------

    def describe(self) -> str:
        """Human-readable description of the board configuration."""
        if self.board_type == 'charuco':
            return (f"ChArUco {self.board_cols}x{self.board_rows} "
                    f"({self.square_size_m*1000:.0f}mm squares, "
                    f"{self.marker_size_m*1000:.0f}mm markers)")
        else:
            return (f"Checkerboard {self.inner_cols}x{self.inner_rows} "
                    f"inner corners ({self.square_size_m*1000:.0f}mm squares)")
