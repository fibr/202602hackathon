"""Tests for the grasp planner waypoint generation."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from planner.grasp_planner import GraspPlanner, Waypoint, MotionType, GripperAction


@pytest.fixture
def planner():
    return GraspPlanner(
        safe_z=300.0,
        approach_offset_z=100.0,
        place_position=(300.0, 0.0, 50.0),
    )


class TestWaypointGeneration:
    def test_waypoint_count(self, planner):
        center = np.array([200.0, 100.0, 13.5])
        axis = np.array([1.0, 0.0, 0.0])
        wps = planner.plan(center, axis)
        assert len(wps) == 9, f"Expected 9 waypoints, got {len(wps)}"

    def test_waypoint_labels(self, planner):
        center = np.array([200.0, 100.0, 13.5])
        axis = np.array([1.0, 0.0, 0.0])
        wps = planner.plan(center, axis)
        labels = [w.label for w in wps]
        assert labels == [
            'PRE_GRASP', 'GRASP_DESCEND', 'GRASP_CLOSE', 'LIFT',
            'MOVE_TO_PLACE', 'REORIENT', 'PLACE_DESCEND', 'RELEASE', 'RETRACT'
        ]

    def test_gripper_sequence(self, planner):
        center = np.array([200.0, 100.0, 13.5])
        axis = np.array([1.0, 0.0, 0.0])
        wps = planner.plan(center, axis)

        gripper_actions = [(w.label, w.gripper) for w in wps if w.gripper != GripperAction.NONE]
        assert gripper_actions == [
            ('PRE_GRASP', GripperAction.OPEN),
            ('GRASP_CLOSE', GripperAction.CLOSE),
            ('RELEASE', GripperAction.OPEN),
        ]

    def test_pre_grasp_above_rod(self, planner):
        center = np.array([200.0, 100.0, 13.5])
        axis = np.array([1.0, 0.0, 0.0])
        wps = planner.plan(center, axis)
        pre = wps[0]
        assert pre.x == pytest.approx(200.0)
        assert pre.y == pytest.approx(100.0)
        assert pre.z == pytest.approx(13.5 + 100.0)  # rod_z + approach_offset

    def test_grasp_at_rod_height(self, planner):
        center = np.array([200.0, 100.0, 13.5])
        axis = np.array([1.0, 0.0, 0.0])
        wps = planner.plan(center, axis)
        descend = wps[1]
        assert descend.z == pytest.approx(13.5)

    def test_lift_to_safe_z(self, planner):
        center = np.array([200.0, 100.0, 13.5])
        axis = np.array([1.0, 0.0, 0.0])
        wps = planner.plan(center, axis)
        lift = wps[3]
        assert lift.z == pytest.approx(300.0)

    def test_place_position(self, planner):
        center = np.array([200.0, 100.0, 13.5])
        axis = np.array([1.0, 0.0, 0.0])
        wps = planner.plan(center, axis)
        place = wps[6]  # PLACE_DESCEND
        assert place.x == pytest.approx(300.0)
        assert place.y == pytest.approx(0.0)
        assert place.z == pytest.approx(50.0)

    def test_motion_types(self, planner):
        center = np.array([200.0, 100.0, 13.5])
        axis = np.array([1.0, 0.0, 0.0])
        wps = planner.plan(center, axis)
        # Pre-grasp and move-to-place use joint (fast repositioning)
        assert wps[0].motion == MotionType.JOINT   # PRE_GRASP
        assert wps[1].motion == MotionType.LINEAR   # GRASP_DESCEND
        assert wps[4].motion == MotionType.JOINT    # MOVE_TO_PLACE


class TestGraspOrientation:
    def test_rz_for_x_axis_rod(self, planner):
        """Rod along X axis -> grasp_rz = atan2(0, 1) = 0 degrees."""
        center = np.array([200.0, 100.0, 13.5])
        axis = np.array([1.0, 0.0, 0.0])
        wps = planner.plan(center, axis)
        assert wps[0].rz == pytest.approx(0.0, abs=0.1)

    def test_rz_for_y_axis_rod(self, planner):
        """Rod along Y axis -> grasp_rz = atan2(1, 0) = 90 degrees."""
        center = np.array([200.0, 100.0, 13.5])
        axis = np.array([0.0, 1.0, 0.0])
        wps = planner.plan(center, axis)
        assert wps[0].rz == pytest.approx(90.0, abs=0.1)

    def test_rz_for_45_deg_rod(self, planner):
        """Rod at 45 degrees."""
        center = np.array([200.0, 100.0, 13.5])
        axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
        wps = planner.plan(center, axis)
        assert wps[0].rz == pytest.approx(45.0, abs=0.1)

    def test_gripper_points_down(self, planner):
        center = np.array([200.0, 100.0, 13.5])
        axis = np.array([1.0, 0.0, 0.0])
        wps = planner.plan(center, axis)
        assert wps[0].rx == pytest.approx(180.0)
        assert wps[0].ry == pytest.approx(0.0)

    def test_reorient_tilts_90(self, planner):
        center = np.array([200.0, 100.0, 13.5])
        axis = np.array([1.0, 0.0, 0.0])
        wps = planner.plan(center, axis)
        reorient = wps[5]  # REORIENT
        assert reorient.rx == pytest.approx(90.0)  # 180 - 90
