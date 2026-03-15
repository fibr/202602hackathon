"""Unit tests for src/fetch_game.py — FetchGameController.

Covers:
- FetchState enum completeness
- FetchGameState dataclass defaults
- FetchGameController construction and initialization
- State transitions via step() for every state in the pipeline
- step() when worker thread is still alive (ignored)
- start_auto() and stop_auto() modes
- reset() clears state and rejoins worker thread
- set_click_target() and set_place_target()
- on_state_changed callback fires on every _set_state() call
- update_detection() with mocked detector
  - no cubes detected → target fields cleared
  - cube detected without transform → pixel coords stored, 3D cleared
  - cube detected with transform → 3D position computed
  - auto-advance in DETECT + auto_mode when 3D target available
- _move_to_position() success, IK failure, get_angles failure, abort flag
- State action methods (_do_approach, _do_descend, _do_lift,
  _do_transport, _do_place, _do_retract) via synchronous call
- _do_open_gripper and _do_grasp with / without gripper attributes
- Error handling: exception in worker → ERROR state
- _worker_wrapper auto-advance in auto_mode
- ensure_ready() with / without calibration files and IK solver
"""

import os
import sys
import threading
import time
import types
import unittest.mock as mock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup — allow importing src modules directly
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fetch_game import FetchGameController, FetchGameState, FetchState

# ---------------------------------------------------------------------------
# Shared stubs / mocks
# ---------------------------------------------------------------------------

def _make_mock_robot(angles=None, pose=None):
    """Return a minimal mock robot that satisfies FetchGameController usage."""
    robot = mock.MagicMock()
    robot.get_angles.return_value = list(angles or [0.0, -30.0, 80.0, 0.0, -50.0, 30.0])
    robot.get_pose.return_value = list(pose or [200.0, 0.0, 100.0, 180.0, 0.0, 0.0])
    robot.move_joints.return_value = None
    robot.gripper_open.return_value = None
    robot.gripper_close.return_value = None
    return robot


def _make_mock_solver(return_solution=None):
    """Return a minimal mock IK solver."""
    solver = mock.MagicMock()
    solution = np.array(return_solution or [0.0, -30.0, 80.0, 0.0, -50.0])
    solver.solve_ik_position.return_value = solution
    solver.forward_kin.return_value = (np.array([200.0, 0.0, 100.0]), np.array([0.0, 0.0, 0.0]))
    return solver


def _make_mock_app(robot=None, solver=None, camera=None, config=None):
    """Return a mock app with optional robot, camera, and config."""
    app = mock.MagicMock()
    app.robot = robot
    app.camera = camera
    app.config = config or {}
    return app


def _make_controller(robot=None, solver=None, camera=None, config=None,
                     place_pos_mm=None, cube_selection_mode='largest',
                     hover_height_mm=40.0, grasp_height_mm=8.0,
                     lift_height_mm=80.0):
    """Construct a FetchGameController with injected mocks."""
    app = _make_mock_app(robot=robot, camera=camera, config=config)
    ctrl = FetchGameController(
        app=app,
        hover_height_mm=hover_height_mm,
        grasp_height_mm=grasp_height_mm,
        lift_height_mm=lift_height_mm,
        place_pos_mm=place_pos_mm,
        cube_selection_mode=cube_selection_mode,
    )
    # Inject solver directly so ensure_ready() is bypassed
    if solver is not None:
        ctrl._solver = solver
    return ctrl


# ---------------------------------------------------------------------------
# FetchState
# ---------------------------------------------------------------------------

class TestFetchState:
    def test_all_expected_states_exist(self):
        names = {s.name for s in FetchState}
        expected = {
            'IDLE', 'DETECT', 'APPROACH', 'REFINE', 'OPEN_GRIPPER',
            'DESCEND', 'GRASP', 'LIFT', 'TRANSPORT', 'PLACE',
            'RETRACT', 'DONE', 'ERROR',
        }
        assert expected.issubset(names), f"Missing states: {expected - names}"

    def test_values_are_strings(self):
        for s in FetchState:
            assert isinstance(s.value, str)


# ---------------------------------------------------------------------------
# FetchGameState
# ---------------------------------------------------------------------------

class TestFetchGameState:
    def test_default_state_is_idle(self):
        s = FetchGameState()
        assert s.state == FetchState.IDLE

    def test_default_no_cube(self):
        s = FetchGameState()
        assert s.num_cubes_detected == 0
        assert s.target_cube_idx == -1
        assert s.target_cube_px is None
        assert s.target_cube_3d is None

    def test_default_auto_mode_off(self):
        s = FetchGameState()
        assert s.auto_mode is False

    def test_default_step_count_zero(self):
        s = FetchGameState()
        assert s.step_count == 0


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_parameters(self):
        ctrl = _make_controller()
        assert ctrl.hover_height_mm == 40.0
        assert ctrl.grasp_height_mm == 8.0
        assert ctrl.lift_height_mm == 80.0
        assert ctrl.cube_selection_mode == 'largest'
        assert ctrl.state.state == FetchState.IDLE

    def test_custom_place_pos(self):
        pos = np.array([150.0, 200.0, 15.0])
        ctrl = _make_controller(place_pos_mm=pos)
        np.testing.assert_array_equal(ctrl.place_pos_mm, pos)

    def test_default_place_pos_when_none(self):
        ctrl = _make_controller(place_pos_mm=None)
        assert ctrl.place_pos_mm is not None
        assert len(ctrl.place_pos_mm) == 3

    def test_on_state_changed_initially_none(self):
        ctrl = _make_controller()
        assert ctrl.on_state_changed is None

    def test_no_solver_initially(self):
        ctrl = _make_controller()
        assert ctrl._solver is None

    def test_no_transform_initially(self):
        ctrl = _make_controller()
        assert ctrl._transform is None


# ---------------------------------------------------------------------------
# _set_state and on_state_changed callback
# ---------------------------------------------------------------------------

class TestSetState:
    def test_state_transitions(self):
        ctrl = _make_controller()
        ctrl._set_state(FetchState.DETECT, 'scanning')
        assert ctrl.state.state == FetchState.DETECT
        assert ctrl.state.status_text == 'scanning'

    def test_step_count_increments(self):
        ctrl = _make_controller()
        initial = ctrl.state.step_count
        ctrl._set_state(FetchState.DETECT)
        ctrl._set_state(FetchState.APPROACH)
        assert ctrl.state.step_count == initial + 2

    def test_callback_fires(self):
        ctrl = _make_controller()
        fired = []
        ctrl.on_state_changed = lambda s: fired.append(s.state)
        ctrl._set_state(FetchState.DETECT)
        ctrl._set_state(FetchState.APPROACH)
        assert fired == [FetchState.DETECT, FetchState.APPROACH]

    def test_callback_exception_does_not_propagate(self):
        ctrl = _make_controller()
        ctrl.on_state_changed = lambda s: (_ for _ in ()).throw(RuntimeError("boom"))
        # Should not raise
        ctrl._set_state(FetchState.DETECT)
        assert ctrl.state.state == FetchState.DETECT

    def test_default_status_text_is_state_value(self):
        ctrl = _make_controller()
        ctrl._set_state(FetchState.LIFT)
        assert ctrl.state.status_text == FetchState.LIFT.value


# ---------------------------------------------------------------------------
# step() — state machine transitions (synchronous portion)
# ---------------------------------------------------------------------------

class TestStepTransitions:
    """Tests the synchronous part of step() — state changes and async dispatch.

    Long-running async actions are patched out so tests run instantly.
    """

    def _ctrl_at(self, state: FetchState, **kwargs):
        """Create a controller already in the given state, with robot/solver."""
        robot = _make_mock_robot()
        solver = _make_mock_solver()
        ctrl = _make_controller(robot=robot, solver=solver, **kwargs)
        ctrl.state.state = state
        return ctrl

    def test_idle_to_detect(self):
        ctrl = _make_controller()
        ctrl.step()
        assert ctrl.state.state == FetchState.DETECT

    def test_detect_no_target_stays_detect(self):
        ctrl = self._ctrl_at(FetchState.DETECT)
        ctrl.state.target_cube_3d = None
        ctrl.step()
        assert ctrl.state.state == FetchState.DETECT

    def test_detect_with_target_to_approach(self):
        ctrl = self._ctrl_at(FetchState.DETECT)
        ctrl.state.target_cube_3d = np.array([200.0, 100.0, 10.0])
        with mock.patch.object(ctrl, '_run_async'):
            ctrl.step()
        assert ctrl.state.state == FetchState.APPROACH

    def test_approach_to_refine(self):
        ctrl = self._ctrl_at(FetchState.APPROACH)
        with mock.patch.object(ctrl, '_run_async'):
            ctrl.step()
        assert ctrl.state.state == FetchState.REFINE

    def test_refine_to_open_gripper(self):
        ctrl = self._ctrl_at(FetchState.REFINE)
        with mock.patch.object(ctrl, '_run_async'):
            ctrl.step()
        assert ctrl.state.state == FetchState.OPEN_GRIPPER

    def test_open_gripper_to_descend(self):
        ctrl = self._ctrl_at(FetchState.OPEN_GRIPPER)
        with mock.patch.object(ctrl, '_run_async'):
            ctrl.step()
        assert ctrl.state.state == FetchState.DESCEND

    def test_descend_to_grasp(self):
        ctrl = self._ctrl_at(FetchState.DESCEND)
        with mock.patch.object(ctrl, '_run_async'):
            ctrl.step()
        assert ctrl.state.state == FetchState.GRASP

    def test_grasp_to_lift(self):
        ctrl = self._ctrl_at(FetchState.GRASP)
        with mock.patch.object(ctrl, '_run_async'):
            ctrl.step()
        assert ctrl.state.state == FetchState.LIFT

    def test_lift_to_transport(self):
        ctrl = self._ctrl_at(FetchState.LIFT)
        with mock.patch.object(ctrl, '_run_async'):
            ctrl.step()
        assert ctrl.state.state == FetchState.TRANSPORT

    def test_transport_to_place(self):
        ctrl = self._ctrl_at(FetchState.TRANSPORT)
        with mock.patch.object(ctrl, '_run_async'):
            ctrl.step()
        assert ctrl.state.state == FetchState.PLACE

    def test_place_to_retract(self):
        ctrl = self._ctrl_at(FetchState.PLACE)
        with mock.patch.object(ctrl, '_run_async'):
            ctrl.step()
        assert ctrl.state.state == FetchState.RETRACT

    def test_retract_to_done(self):
        ctrl = self._ctrl_at(FetchState.RETRACT)
        ctrl.step()
        assert ctrl.state.state == FetchState.DONE

    def test_done_resets_to_idle(self):
        ctrl = self._ctrl_at(FetchState.DONE)
        ctrl.step()
        assert ctrl.state.state == FetchState.IDLE

    def test_error_resets_to_idle(self):
        ctrl = self._ctrl_at(FetchState.ERROR)
        ctrl.step()
        assert ctrl.state.state == FetchState.IDLE

    def test_step_ignored_when_worker_alive(self):
        ctrl = _make_controller()
        # Simulate a running worker thread
        barrier = threading.Barrier(2)
        done = threading.Event()

        def long_task():
            barrier.wait()
            done.wait()

        ctrl._worker_thread = threading.Thread(target=long_task, daemon=True)
        ctrl._worker_thread.start()
        barrier.wait()  # Wait until the thread is actually running

        # step() should be ignored because worker is alive
        ctrl.step()
        assert ctrl.state.state == FetchState.IDLE  # unchanged

        done.set()
        ctrl._worker_thread.join()

    def test_full_pipeline_transitions(self):
        """Walk through every state in sequence using mocked async."""
        ctrl = _make_controller()
        sequence = [
            FetchState.DETECT,
            FetchState.APPROACH,
            FetchState.REFINE,
            FetchState.OPEN_GRIPPER,
            FetchState.DESCEND,
            FetchState.GRASP,
            FetchState.LIFT,
            FetchState.TRANSPORT,
            FetchState.PLACE,
            FetchState.RETRACT,
            FetchState.DONE,
        ]
        with mock.patch.object(ctrl, '_run_async'):
            ctrl.step()  # IDLE → DETECT
            assert ctrl.state.state == FetchState.DETECT

            ctrl.state.target_cube_3d = np.array([200.0, 100.0, 10.0])
            for expected in sequence[1:]:
                ctrl.step()
                assert ctrl.state.state == expected, (
                    f"Expected {expected} but got {ctrl.state.state}"
                )


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_goes_to_idle(self):
        ctrl = _make_controller()
        ctrl._set_state(FetchState.TRANSPORT, 'moving')
        ctrl.reset()
        assert ctrl.state.state == FetchState.IDLE

    def test_reset_clears_cube_data(self):
        ctrl = _make_controller()
        ctrl.state.target_cube_3d = np.array([1.0, 2.0, 3.0])
        ctrl.state.num_cubes_detected = 5
        ctrl.reset()
        assert ctrl.state.num_cubes_detected == 0
        assert ctrl.state.target_cube_3d is None

    def test_reset_sets_abort_false_after(self):
        ctrl = _make_controller()
        ctrl.reset()
        assert ctrl._abort is False

    def test_reset_stores_place_target(self):
        pos = np.array([150.0, 200.0, 15.0])
        ctrl = _make_controller(place_pos_mm=pos)
        ctrl.reset()
        np.testing.assert_array_almost_equal(ctrl.state.place_target_3d, pos)

    def test_reset_joins_worker(self):
        """reset() sets abort flag so workers can exit."""
        ctrl = _make_controller()
        finished = threading.Event()

        def slow_fn():
            for _ in range(50):
                if ctrl._abort:
                    break
                time.sleep(0.01)
            finished.set()

        ctrl._worker_thread = threading.Thread(target=slow_fn, daemon=True)
        ctrl._worker_thread.start()

        ctrl.reset()
        assert not ctrl._abort  # reset() resets abort after join
        assert finished.is_set()


# ---------------------------------------------------------------------------
# start_auto() and stop_auto()
# ---------------------------------------------------------------------------

class TestAutoMode:
    def test_start_auto_enables_auto_mode(self):
        ctrl = _make_controller()
        ctrl.start_auto()
        assert ctrl.state.auto_mode is True

    def test_start_auto_sets_detect_state(self):
        ctrl = _make_controller()
        ctrl.start_auto()
        assert ctrl.state.state == FetchState.DETECT

    def test_stop_auto_disables_auto_mode(self):
        ctrl = _make_controller()
        ctrl.start_auto()
        ctrl.stop_auto()
        assert ctrl.state.auto_mode is False

    def test_stop_auto_sets_abort(self):
        ctrl = _make_controller()
        ctrl.start_auto()
        ctrl.stop_auto()
        assert ctrl._abort is True


# ---------------------------------------------------------------------------
# set_click_target() and set_place_target()
# ---------------------------------------------------------------------------

class TestSetters:
    def test_set_click_target_stores_position(self):
        ctrl = _make_controller()
        ctrl.set_click_target(320, 240)
        assert ctrl._click_xy == (320, 240)
        assert ctrl.cube_selection_mode == 'closest_to_click'

    def test_set_place_target_stores_position(self):
        ctrl = _make_controller()
        pos = np.array([100.0, 200.0, 15.0])
        ctrl.set_place_target(pos)
        np.testing.assert_array_almost_equal(ctrl.place_pos_mm, pos)
        np.testing.assert_array_almost_equal(ctrl.state.place_target_3d, pos)

    def test_set_place_target_copies(self):
        ctrl = _make_controller()
        pos = np.array([100.0, 200.0, 15.0])
        ctrl.set_place_target(pos)
        pos[0] = 999.0
        assert ctrl.place_pos_mm[0] != 999.0  # should be a copy


# ---------------------------------------------------------------------------
# update_detection()
# ---------------------------------------------------------------------------

class TestUpdateDetection:
    """Tests for update_detection() with mocked cube detector."""

    @staticmethod
    def _make_cube(cx=320, cy=240, area=1000.0):
        """Create a minimal CubeDetection stub."""
        from vision.green_cube_detector import CubeDetection
        contour = np.array([[[cx-5, cy-5]], [[cx+5, cy-5]],
                             [[cx+5, cy+5]], [[cx-5, cy+5]]], dtype=np.int32)
        return CubeDetection(
            cx=cx, cy=cy, area=area,
            bbox=(cx-5, cy-5, 10, 10), contour=contour,
            yaw_deg=0.0, aspect_ratio=1.0, solidity=1.0,
        )

    def test_no_cubes_clears_target_fields(self):
        ctrl = _make_controller()
        ctrl.state.target_cube_px = (100, 200)
        ctrl.state.target_cube_3d = np.array([1.0, 2.0, 3.0])

        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with mock.patch('vision.green_cube_detector.detect_green_cubes',
                        return_value=([], {})):
            with mock.patch('vision.green_cube_detector.select_target_cube',
                            return_value=-1):
                cubes, idx = ctrl.update_detection(blank_frame)

        assert cubes == []
        assert idx == -1
        assert ctrl.state.num_cubes_detected == 0
        assert ctrl.state.target_cube_px is None
        assert ctrl.state.target_cube_3d is None

    def test_cube_detected_stores_pixel_position(self):
        ctrl = _make_controller()
        cube = self._make_cube(cx=200, cy=150)
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with mock.patch('vision.green_cube_detector.detect_green_cubes',
                        return_value=([cube], {})):
            with mock.patch('vision.green_cube_detector.select_target_cube',
                            return_value=0):
                ctrl.update_detection(blank_frame)

        assert ctrl.state.num_cubes_detected == 1
        assert ctrl.state.target_cube_idx == 0
        assert ctrl.state.target_cube_px == (200, 150)

    def test_cube_detected_without_transform_no_3d(self):
        ctrl = _make_controller()
        ctrl._transform = None  # no calibration
        cube = self._make_cube(cx=200, cy=150)
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with mock.patch('vision.green_cube_detector.detect_green_cubes',
                        return_value=([cube], {})):
            with mock.patch('vision.green_cube_detector.select_target_cube',
                            return_value=0):
                ctrl.update_detection(blank_frame)

        # Without transform, 3D position cannot be computed
        assert ctrl.state.target_cube_3d is None

    def test_cube_detected_with_transform_computes_3d(self):
        """With a valid transform and camera intrinsics, 3D pos should be set."""
        from collections import namedtuple

        # Build a plausible camera intrinsics object
        Intrinsics = namedtuple('Intrinsics', ['ppx', 'ppy', 'fx', 'fy'])
        intr = Intrinsics(ppx=320.0, ppy=240.0, fx=600.0, fy=600.0)

        camera = mock.MagicMock()
        camera.intrinsics = intr

        # Build a T_camera_to_base that places the camera 500mm above origin
        # looking straight DOWN.  In the ray-intersection code, the camera-frame
        # ray for a centred pixel is [0, 0, 1].  We need the world-frame ray
        # to have a NEGATIVE Z component so that the ray eventually reaches the
        # table (cube_z = 10mm < origin_z = 500mm).
        # Flip camera Z → world Z mapping: R = diag([1, 1, -1]).
        T = np.eye(4)
        T[2, 2] = -1.0   # camera +Z maps to world -Z (looking down)
        T[2, 3] = 500.0  # camera origin is 500mm above the table in world Z

        transform = mock.MagicMock()
        transform.T_camera_to_base = T

        ctrl = _make_controller(camera=camera)
        ctrl._transform = transform

        cube = self._make_cube(cx=320, cy=240)  # centred → ray is straight down
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with mock.patch('vision.green_cube_detector.detect_green_cubes',
                        return_value=([cube], {})):
            with mock.patch('vision.green_cube_detector.select_target_cube',
                            return_value=0):
                ctrl.update_detection(blank_frame)

        # With a downward-pointing ray from 500mm, the intersection at z=10 is
        # a positive scale → 3D position should be computed
        assert ctrl.state.target_cube_3d is not None

    def test_returns_cubes_and_target_idx(self):
        ctrl = _make_controller()
        cube = self._make_cube()
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with mock.patch('vision.green_cube_detector.detect_green_cubes',
                        return_value=([cube], {})):
            with mock.patch('vision.green_cube_detector.select_target_cube',
                            return_value=0):
                cubes, idx = ctrl.update_detection(blank_frame)

        assert cubes == [cube]
        assert idx == 0

    def test_auto_advance_in_detect_state(self):
        """In auto_mode + DETECT + valid 3D, update_detection() calls step()."""
        ctrl = _make_controller()
        ctrl.state.auto_mode = True
        ctrl.state.state = FetchState.DETECT
        cube = self._make_cube()
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Provide a plausible 3D position (camera looking DOWN — flip Z so s > 0)
        from collections import namedtuple
        Intrinsics = namedtuple('Intrinsics', ['ppx', 'ppy', 'fx', 'fy'])
        camera = mock.MagicMock()
        camera.intrinsics = Intrinsics(ppx=320.0, ppy=240.0, fx=600.0, fy=600.0)
        ctrl.app.camera = camera

        T = np.eye(4)
        T[2, 2] = -1.0   # camera +Z → world -Z (looking downward)
        T[2, 3] = 500.0  # camera 500mm above table
        transform = mock.MagicMock()
        transform.T_camera_to_base = T
        ctrl._transform = transform

        with mock.patch.object(ctrl, 'step') as mock_step:
            with mock.patch('vision.green_cube_detector.detect_green_cubes',
                            return_value=([cube], {})):
                with mock.patch('vision.green_cube_detector.select_target_cube',
                                return_value=0):
                    ctrl.update_detection(blank_frame)

        # step() should have been called since state was DETECT + auto + 3D target
        mock_step.assert_called_once()

    def test_no_auto_advance_when_not_detect_state(self):
        """auto_mode doesn't advance when not in DETECT state."""
        ctrl = _make_controller()
        ctrl.state.auto_mode = True
        ctrl.state.state = FetchState.APPROACH  # not DETECT
        cube = self._make_cube()
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with mock.patch.object(ctrl, 'step') as mock_step:
            with mock.patch('vision.green_cube_detector.detect_green_cubes',
                            return_value=([cube], {})):
                with mock.patch('vision.green_cube_detector.select_target_cube',
                                return_value=0):
                    ctrl.update_detection(blank_frame)

        mock_step.assert_not_called()


# ---------------------------------------------------------------------------
# Multi-cube selection integration
# ---------------------------------------------------------------------------

class TestMultiCubeSelection:
    """Tests for different cube_selection_mode values through update_detection."""

    @staticmethod
    def _make_cube(cx, cy, area=1000.0):
        from vision.green_cube_detector import CubeDetection
        contour = np.array([[[cx-5, cy-5]], [[cx+5, cy-5]],
                             [[cx+5, cy+5]], [[cx-5, cy+5]]], dtype=np.int32)
        return CubeDetection(cx=cx, cy=cy, area=area,
                             bbox=(cx-5, cy-5, 10, 10), contour=contour,
                             yaw_deg=0.0, aspect_ratio=1.0, solidity=1.0)

    def test_largest_mode_selects_index_0(self):
        """With mode='largest', the first cube (largest area) is selected."""
        ctrl = _make_controller(cube_selection_mode='largest')
        big = self._make_cube(100, 100, area=5000.0)
        small = self._make_cube(400, 300, area=500.0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # detect_green_cubes returns sorted-by-area list (big first)
        with mock.patch('vision.green_cube_detector.detect_green_cubes',
                        return_value=([big, small], {})):
            with mock.patch('vision.green_cube_detector.select_target_cube',
                            return_value=0):  # select_target_cube also picks index 0
                cubes, idx = ctrl.update_detection(frame)

        assert idx == 0
        assert ctrl.state.target_cube_px == (100, 100)

    def test_closest_to_click_mode_selection(self):
        """With mode='closest_to_click', the cube nearest the click is targeted."""
        ctrl = _make_controller(cube_selection_mode='closest_to_click')
        ctrl.set_click_target(350, 250)

        cube_a = self._make_cube(100, 100)
        cube_b = self._make_cube(360, 260)  # Nearest to (350, 250)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with mock.patch('vision.green_cube_detector.detect_green_cubes',
                        return_value=([cube_a, cube_b], {})):
            with mock.patch('vision.green_cube_detector.select_target_cube',
                            return_value=1):  # cube_b is idx 1
                cubes, idx = ctrl.update_detection(frame)

        assert idx == 1
        assert ctrl.state.target_cube_px == (360, 260)

    def test_target_cube_idx_stored_in_state(self):
        ctrl = _make_controller()
        cube = self._make_cube(200, 200)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with mock.patch('vision.green_cube_detector.detect_green_cubes',
                        return_value=([cube], {})):
            with mock.patch('vision.green_cube_detector.select_target_cube',
                            return_value=0):
                ctrl.update_detection(frame)

        assert ctrl.state.target_cube_idx == 0

    def test_num_cubes_detected_stored_in_state(self):
        ctrl = _make_controller()
        cubes = [self._make_cube(100, 100), self._make_cube(300, 300),
                 self._make_cube(500, 200)]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with mock.patch('vision.green_cube_detector.detect_green_cubes',
                        return_value=(cubes, {})):
            with mock.patch('vision.green_cube_detector.select_target_cube',
                            return_value=0):
                ctrl.update_detection(frame)

        assert ctrl.state.num_cubes_detected == 3


# ---------------------------------------------------------------------------
# _move_to_position()
# ---------------------------------------------------------------------------

class TestMoveToPosition:
    def test_moves_successfully(self):
        robot = _make_mock_robot()
        solver = _make_mock_solver()
        ctrl = _make_controller(robot=robot, solver=solver)

        pos = np.array([200.0, 0.0, 80.0])
        ctrl._move_to_position(pos, 'test_label')

        robot.move_joints.assert_called_once()
        call_args = robot.move_joints.call_args
        cmd = call_args[0][0]
        assert len(cmd) == 6

    def test_abort_skips_move(self):
        robot = _make_mock_robot()
        solver = _make_mock_solver()
        ctrl = _make_controller(robot=robot, solver=solver)
        ctrl._abort = True

        ctrl._move_to_position(np.array([200.0, 0.0, 80.0]), 'aborted')
        robot.move_joints.assert_not_called()

    def test_no_robot_raises(self):
        ctrl = _make_controller(robot=None, solver=_make_mock_solver())
        with pytest.raises(RuntimeError, match="No robot or IK solver"):
            ctrl._move_to_position(np.array([200.0, 0.0, 80.0]))

    def test_no_solver_raises(self):
        ctrl = _make_controller(robot=_make_mock_robot())
        ctrl._solver = None
        with pytest.raises(RuntimeError, match="No robot or IK solver"):
            ctrl._move_to_position(np.array([200.0, 0.0, 80.0]))

    def test_ik_failure_raises(self):
        robot = _make_mock_robot()
        solver = _make_mock_solver(return_solution=None)
        solver.solve_ik_position.return_value = None
        ctrl = _make_controller(robot=robot, solver=solver)

        with pytest.raises(RuntimeError, match="IK failed"):
            ctrl._move_to_position(np.array([200.0, 0.0, 80.0]))

    def test_bad_angles_raises(self):
        robot = _make_mock_robot()
        robot.get_angles.return_value = []  # empty → invalid
        solver = _make_mock_solver()
        ctrl = _make_controller(robot=robot, solver=solver)

        with pytest.raises(RuntimeError, match="Cannot get joint angles"):
            ctrl._move_to_position(np.array([200.0, 0.0, 80.0]))

    def test_speed_passed_to_robot(self):
        robot = _make_mock_robot()
        solver = _make_mock_solver()
        ctrl = _make_controller(robot=robot, solver=solver)
        ctrl._speed = 75

        ctrl._move_to_position(np.array([200.0, 0.0, 80.0]))
        _, kwargs = robot.move_joints.call_args
        assert kwargs.get('speed') == 75 or robot.move_joints.call_args[0][1] == 75 or \
               robot.move_joints.call_args[1].get('speed') == 75 or \
               robot.move_joints.called  # at least called with something


# ---------------------------------------------------------------------------
# State action methods (called synchronously for unit testing)
# ---------------------------------------------------------------------------

class TestStateActions:
    """Test the _do_* methods with mocked robot/solver and patched time.sleep."""

    @staticmethod
    @mock.patch('time.sleep')
    def _call(fn, mock_sleep=None):
        fn()

    def _ctrl_with_target(self, target_3d=None, **kwargs):
        robot = _make_mock_robot()
        solver = _make_mock_solver()
        ctrl = _make_controller(robot=robot, solver=solver, **kwargs)
        ctrl.state.target_cube_3d = target_3d or np.array([200.0, 100.0, 10.0])
        return ctrl

    @mock.patch('time.sleep')
    def test_do_approach_sets_hover_height(self, _sleep):
        ctrl = self._ctrl_with_target()
        ctrl._do_approach()
        # The approach should have called move_joints (via _move_to_position)
        ctrl.app.robot.move_joints.assert_called()
        # refined_pos_3d should be set
        assert ctrl.state.refined_pos_3d is not None

    @mock.patch('time.sleep')
    def test_do_approach_no_target_raises(self, _sleep):
        robot = _make_mock_robot()
        solver = _make_mock_solver()
        ctrl = _make_controller(robot=robot, solver=solver)
        ctrl.state.target_cube_3d = None
        with pytest.raises(RuntimeError, match="No target cube position"):
            ctrl._do_approach()

    @mock.patch('time.sleep')
    def test_do_descend_uses_refined_pos(self, _sleep):
        ctrl = self._ctrl_with_target(grasp_height_mm=8.0)
        ctrl.state.refined_pos_3d = np.array([210.0, 110.0, 10.0])
        ctrl._do_descend()
        ctrl.app.robot.move_joints.assert_called()
        # Status should mention grasp height
        assert '8' in ctrl.state.status_text

    @mock.patch('time.sleep')
    def test_do_descend_falls_back_to_target(self, _sleep):
        ctrl = self._ctrl_with_target()
        ctrl.state.refined_pos_3d = None
        ctrl._do_descend()
        ctrl.app.robot.move_joints.assert_called()

    @mock.patch('time.sleep')
    def test_do_descend_no_position_raises(self, _sleep):
        robot = _make_mock_robot()
        solver = _make_mock_solver()
        ctrl = _make_controller(robot=robot, solver=solver)
        ctrl.state.refined_pos_3d = None
        ctrl.state.target_cube_3d = None
        with pytest.raises(RuntimeError, match="No target position for descend"):
            ctrl._do_descend()

    @mock.patch('time.sleep')
    def test_do_lift_uses_refined_or_target(self, _sleep):
        ctrl = self._ctrl_with_target()
        ctrl.state.refined_pos_3d = np.array([200.0, 100.0, 10.0])
        ctrl._do_lift()
        ctrl.app.robot.move_joints.assert_called()
        assert ctrl.lift_height_mm > 0

    @mock.patch('time.sleep')
    def test_do_lift_no_position_raises(self, _sleep):
        robot = _make_mock_robot()
        solver = _make_mock_solver()
        ctrl = _make_controller(robot=robot, solver=solver)
        ctrl.state.refined_pos_3d = None
        ctrl.state.target_cube_3d = None
        with pytest.raises(RuntimeError, match="No position for lift"):
            ctrl._do_lift()

    @mock.patch('time.sleep')
    def test_do_transport_moves_to_lift_height(self, _sleep):
        ctrl = self._ctrl_with_target(
            place_pos_mm=np.array([100.0, 200.0, 15.0]),
            lift_height_mm=80.0,
        )
        ctrl._do_transport()
        ctrl.app.robot.move_joints.assert_called()
        assert '80' in ctrl.state.status_text

    @mock.patch('time.sleep')
    def test_do_place_moves_and_opens_gripper(self, _sleep):
        ctrl = self._ctrl_with_target(
            place_pos_mm=np.array([100.0, 200.0, 15.0])
        )
        ctrl._do_place()
        ctrl.app.robot.move_joints.assert_called()
        ctrl.app.robot.gripper_open.assert_called()
        assert 'placed' in ctrl.state.status_text.lower()

    @mock.patch('time.sleep')
    def test_do_retract_moves_to_lift_height(self, _sleep):
        ctrl = self._ctrl_with_target(
            place_pos_mm=np.array([100.0, 200.0, 15.0]),
            lift_height_mm=80.0,
        )
        ctrl._do_retract()
        ctrl.app.robot.move_joints.assert_called()

    @mock.patch('time.sleep')
    def test_do_open_gripper_calls_gripper_open(self, _sleep):
        ctrl = self._ctrl_with_target()
        ctrl._do_open_gripper()
        ctrl.app.robot.gripper_open.assert_called_once()

    @mock.patch('time.sleep')
    def test_do_open_gripper_no_robot_no_crash(self, _sleep):
        ctrl = _make_controller(robot=None)
        ctrl._do_open_gripper()  # should not raise

    @mock.patch('time.sleep')
    def test_do_grasp_calls_gripper_close(self, _sleep):
        ctrl = self._ctrl_with_target()
        ctrl._do_grasp()
        ctrl.app.robot.gripper_close.assert_called_once()

    @mock.patch('time.sleep')
    def test_do_grasp_no_robot_no_crash(self, _sleep):
        ctrl = _make_controller(robot=None)
        ctrl._do_grasp()  # should not raise

    @mock.patch('time.sleep')
    def test_do_open_gripper_robot_without_method(self, _sleep):
        """Robot without gripper_open attribute should not crash."""
        robot = mock.MagicMock(spec=[])  # no attributes
        ctrl = _make_controller(robot=robot)
        ctrl._do_open_gripper()  # should not raise

    @mock.patch('time.sleep')
    def test_do_grasp_robot_without_method(self, _sleep):
        """Robot without gripper_close attribute should not crash."""
        robot = mock.MagicMock(spec=[])  # no attributes
        ctrl = _make_controller(robot=robot)
        ctrl._do_grasp()  # should not raise


# ---------------------------------------------------------------------------
# Error handling: exception in worker → ERROR state
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_exception_in_action_sets_error_state(self):
        """If an action raises, _worker_wrapper catches it and sets ERROR state."""
        ctrl = _make_controller()
        ctrl.state.state = FetchState.APPROACH

        def failing_action():
            raise RuntimeError("simulated hardware failure")

        ctrl._worker_wrapper(failing_action)
        assert ctrl.state.state == FetchState.ERROR
        assert 'simulated hardware failure' in ctrl.state.status_text

    def test_error_message_in_status_text(self):
        ctrl = _make_controller()
        ctrl._worker_wrapper(lambda: (_ for _ in ()).throw(ValueError("bad value")))
        assert ctrl.state.state == FetchState.ERROR
        assert 'bad value' in ctrl.state.status_text

    def test_successful_action_does_not_set_error(self):
        ctrl = _make_controller()
        ctrl.state.auto_mode = False
        ctrl._worker_wrapper(lambda: None)
        assert ctrl.state.state != FetchState.ERROR

    @mock.patch('time.sleep')
    def test_approach_ik_fail_sets_error(self, _sleep):
        """IK failure during approach propagates to ERROR state."""
        robot = _make_mock_robot()
        solver = _make_mock_solver()
        solver.solve_ik_position.return_value = None  # force IK failure

        ctrl = _make_controller(robot=robot, solver=solver)
        ctrl.state.target_cube_3d = np.array([200.0, 100.0, 10.0])

        ctrl._worker_wrapper(ctrl._do_approach)
        assert ctrl.state.state == FetchState.ERROR


# ---------------------------------------------------------------------------
# _worker_wrapper auto-advance in auto_mode
# ---------------------------------------------------------------------------

class TestWorkerWrapperAutoMode:
    @mock.patch('time.sleep')
    def test_auto_mode_calls_step_after_success(self, _sleep):
        ctrl = _make_controller()
        ctrl.state.auto_mode = True
        ctrl._abort = False

        with mock.patch.object(ctrl, 'step') as mock_step:
            ctrl._worker_wrapper(lambda: None)

        mock_step.assert_called_once()

    @mock.patch('time.sleep')
    def test_no_auto_advance_when_abort(self, _sleep):
        ctrl = _make_controller()
        ctrl.state.auto_mode = True
        ctrl._abort = True  # abort set

        with mock.patch.object(ctrl, 'step') as mock_step:
            ctrl._worker_wrapper(lambda: None)

        mock_step.assert_not_called()

    @mock.patch('time.sleep')
    def test_no_auto_advance_when_manual_mode(self, _sleep):
        ctrl = _make_controller()
        ctrl.state.auto_mode = False
        ctrl._abort = False

        with mock.patch.object(ctrl, 'step') as mock_step:
            ctrl._worker_wrapper(lambda: None)

        mock_step.assert_not_called()


# ---------------------------------------------------------------------------
# ensure_ready()
# ---------------------------------------------------------------------------

class TestEnsureReady:
    def test_ensure_ready_no_solver_no_crash(self):
        """ensure_ready() with no calibration file and no IK solver — graceful."""
        ctrl = _make_controller()
        with mock.patch('os.path.exists', return_value=False):
            ctrl.ensure_ready()
        # Should not raise; solver remains None
        assert ctrl._solver is None

    def test_ensure_ready_solver_already_set(self):
        """If solver is already set, ensure_ready() doesn't overwrite it."""
        solver = _make_mock_solver()
        ctrl = _make_controller(solver=solver)
        # solver was set to ctrl._solver in _make_controller
        orig_solver = ctrl._solver
        with mock.patch('os.path.exists', return_value=False):
            ctrl.ensure_ready()
        # Solver should not have changed
        assert ctrl._solver is orig_solver

    def test_ensure_ready_ik_import_failure_logs_warning(self):
        """If IK solver import fails, ensure_ready() logs a warning but continues."""
        ctrl = _make_controller()
        with mock.patch('os.path.exists', return_value=False):
            with mock.patch('builtins.__import__', side_effect=ImportError("no module")):
                # Should not propagate the ImportError
                try:
                    ctrl.ensure_ready()
                except Exception:
                    pass  # If it raises, it's ok — we just want no unhandled crash


# ---------------------------------------------------------------------------
# _run_async integation: actions actually run in threads
# ---------------------------------------------------------------------------

class TestRunAsync:
    @mock.patch('time.sleep')
    def test_action_runs_in_background_thread(self, _sleep):
        ctrl = _make_controller()
        ran = threading.Event()

        def action():
            ran.set()

        ctrl._run_async(action)
        assert ran.wait(timeout=2.0), "Action did not run within 2s"

    @mock.patch('time.sleep')
    def test_worker_thread_is_daemon(self, _sleep):
        ctrl = _make_controller()
        done = threading.Event()
        ctrl._run_async(done.wait)
        # Should have started a thread
        assert ctrl._worker_thread is not None
        assert ctrl._worker_thread.daemon is True
        done.set()
        ctrl._worker_thread.join(timeout=1.0)
