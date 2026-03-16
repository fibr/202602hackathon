"""Pinocchio self-collision filter for the SO-ARM101 and Nova5 robots.

When performing primitive-based self-collision checking with Pinocchio's
GeometryModel, links that are directly connected by a joint are always in
physical contact and will always report a collision.  This module provides
the lists of such adjacent (and nearly-adjacent) link pairs so they can be
removed from the active collision-pair set before checking.

Usage — quick example
---------------------
    import pinocchio as pin
    from kinematics.collision_filter import (
        build_collision_model,
        NOVA5_DISABLED_PAIRS,
        SO101_DISABLED_PAIRS,
    )

    # Build geometry model for Nova5
    model, geom_model = build_collision_model(
        urdf_path  = "assets/nova5_robot.urdf",
        srdf_path  = "assets/nova5_robot.srdf",   # optional — filters loaded from here
        mesh_dir   = "assets",
    )
    # …or apply the built-in Python lists directly:
    model, geom_model = build_collision_model(
        urdf_path     = "assets/nova5_robot.urdf",
        disabled_pairs = NOVA5_DISABLED_PAIRS,
    )

    data      = model.createData()
    geom_data = geom_model.createData()

    q = pin.neutral(model)
    in_collision = pin.computeCollisions(model, data, geom_model, geom_data, q, True)

Notes
-----
* "Adjacent" pairs  — links directly connected by a joint.  Always in contact
  regardless of configuration.  Must be disabled.
* "Default" pairs   — links 2 hops apart in the kinematic chain.  For the box/
  cylinder primitives used in this project they always overlap at the joint;
  also disabled.
* Pairs 3+ hops apart are left active so genuine self-collisions are detected.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Iterable, Sequence

import pinocchio as pin

# ---------------------------------------------------------------------------
# Disabled-pair lists (link-name tuples)
# ---------------------------------------------------------------------------

#: Nova5 — adjacent and nearly-adjacent pairs.
#: Source: ``assets/nova5_robot.srdf``
NOVA5_DISABLED_PAIRS: list[tuple[str, str]] = [
    # Adjacent (reason="Adjacent") — directly connected by a joint
    ("dummy_link", "base_link"),
    ("base_link",  "Link1"),
    ("Link1",      "Link2"),
    ("Link2",      "Link3"),
    ("Link3",      "Link4"),
    ("Link4",      "Link5"),
    ("Link5",      "Link6"),
    ("Link6",      "tool_tip"),
    # Nearly-adjacent (reason="Default") — 2 hops, always overlapping
    ("dummy_link", "Link1"),
    ("base_link",  "Link2"),
    ("Link1",      "Link3"),
    ("Link2",      "Link4"),
    ("Link3",      "Link5"),
    ("Link4",      "Link6"),
    ("Link5",      "tool_tip"),
]

#: SO-ARM101 — adjacent and nearly-adjacent pairs.
#: Source: ``assets/so101/so101_new_calib.srdf``
SO101_DISABLED_PAIRS: list[tuple[str, str]] = [
    # Adjacent (reason="Adjacent") — directly connected by a joint
    ("base_link",      "shoulder_link"),
    ("shoulder_link",  "upper_arm_link"),
    ("upper_arm_link", "lower_arm_link"),
    ("lower_arm_link", "wrist_link"),
    ("wrist_link",     "gripper_link"),
    ("gripper_link",   "gripper_frame_link"),
    ("gripper_link",   "moving_jaw_so101_v1_link"),
    # Nearly-adjacent (reason="Default") — 2 hops, always overlapping
    ("base_link",      "upper_arm_link"),
    ("shoulder_link",  "lower_arm_link"),
    ("upper_arm_link", "wrist_link"),
    ("lower_arm_link", "gripper_link"),
    ("wrist_link",     "gripper_frame_link"),
    ("wrist_link",     "moving_jaw_so101_v1_link"),
]


# ---------------------------------------------------------------------------
# SRDF loader
# ---------------------------------------------------------------------------

def load_disabled_pairs_from_srdf(srdf_path: str) -> list[tuple[str, str]]:
    """Parse a MoveIt SRDF file and return all disabled collision pairs.

    Args:
        srdf_path: Absolute or relative path to a ``.srdf`` file.

    Returns:
        List of ``(link1_name, link2_name)`` pairs with collisions disabled.

    Raises:
        FileNotFoundError: if ``srdf_path`` does not exist.
        xml.etree.ElementTree.ParseError: if the file is not valid XML.
    """
    if not os.path.exists(srdf_path):
        raise FileNotFoundError(f"SRDF not found: {srdf_path}")

    tree = ET.parse(srdf_path)
    root = tree.getroot()

    pairs: list[tuple[str, str]] = []
    for elem in root.findall("disable_collisions"):
        link1 = elem.get("link1", "")
        link2 = elem.get("link2", "")
        if link1 and link2:
            pairs.append((link1, link2))
    return pairs


# ---------------------------------------------------------------------------
# Geometry model builder
# ---------------------------------------------------------------------------

def build_collision_model(
    urdf_path: str,
    srdf_path: str | None = None,
    mesh_dir: str | None = None,
    disabled_pairs: Iterable[tuple[str, str]] | None = None,
) -> tuple[pin.Model, pin.GeometryModel]:
    """Build a Pinocchio model + geometry model with self-collision pairs filtered.

    Loads the URDF, builds all primitive collision pairs, then removes the
    pairs in *disabled_pairs* (or those from *srdf_path* if provided).

    Args:
        urdf_path:       Path to the robot URDF.
        srdf_path:       Optional path to the companion SRDF.  If provided,
                         disabled pairs are read from it.  Takes precedence
                         over *disabled_pairs* when both are given.
        mesh_dir:        Directory for resolving ``package://`` or relative
                         mesh filenames.  Pass the repo root or ``assets/``
                         directory.  If ``None``, the URDF directory is used.
        disabled_pairs:  Explicit list of ``(link1, link2)`` pairs to disable.
                         Used when *srdf_path* is ``None``.

    Returns:
        ``(model, collision_geom_model)`` ready for
        ``pin.computeCollisions()``.
    """
    urdf_path = os.path.abspath(urdf_path)
    if mesh_dir is None:
        mesh_dir = os.path.dirname(urdf_path)
    mesh_dir = os.path.abspath(mesh_dir)

    # Kinematic model
    model = pin.buildModelFromUrdf(urdf_path)

    # Geometry model (collision shapes only)
    collision_model = pin.buildGeomFromUrdf(
        model, urdf_path,
        pin.GeometryType.COLLISION,
        package_dirs=[mesh_dir],
    )

    # Add all self-collision pairs
    collision_model.addAllCollisionPairs()

    # Resolve disabled pairs
    if srdf_path is not None:
        pairs_to_remove = load_disabled_pairs_from_srdf(srdf_path)
    elif disabled_pairs is not None:
        pairs_to_remove = list(disabled_pairs)
    else:
        pairs_to_remove = []

    # Build name → geometry-object-index map
    name_to_geom_ids: dict[str, list[int]] = {}
    for idx, go in enumerate(collision_model.geometryObjects):
        parent_name = model.names[go.parentJoint]
        # Some links host the geometry; resolve via parent frame name
        frame_id = go.parentFrame
        if frame_id < model.nframes:
            link_name = model.frames[frame_id].name
        else:
            link_name = parent_name
        name_to_geom_ids.setdefault(link_name, []).append(idx)

    # Remove disabled collision pairs
    removed = 0
    for link1_name, link2_name in pairs_to_remove:
        ids1 = name_to_geom_ids.get(link1_name, [])
        ids2 = name_to_geom_ids.get(link2_name, [])
        for g1 in ids1:
            for g2 in ids2:
                pair = pin.CollisionPair(min(g1, g2), max(g1, g2))
                if collision_model.existCollisionPair(pair):
                    collision_model.removeCollisionPair(pair)
                    removed += 1

    return model, collision_model


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def get_disabled_pairs(robot_name: str) -> list[tuple[str, str]]:
    """Return the built-in disabled-pair list for a named robot.

    Args:
        robot_name: ``"nova5"`` or ``"arm101"`` (case-insensitive).

    Returns:
        List of ``(link1, link2)`` tuples.

    Raises:
        ValueError: if *robot_name* is not recognised.
    """
    name_lower = robot_name.lower()
    if name_lower in ("nova5", "dobot_nova5"):
        return list(NOVA5_DISABLED_PAIRS)
    if name_lower in ("arm101", "so101", "so_arm101"):
        return list(SO101_DISABLED_PAIRS)
    raise ValueError(
        f"Unknown robot '{robot_name}'. "
        "Expected 'nova5' or 'arm101'."
    )


def is_pair_disabled(
    link1: str,
    link2: str,
    disabled_pairs: Sequence[tuple[str, str]],
) -> bool:
    """Check whether a specific link pair is in the disabled list.

    Order-insensitive: ``(A, B)`` and ``(B, A)`` are treated identically.

    Args:
        link1, link2:    Link names to look up.
        disabled_pairs:  List of disabled pairs (e.g. ``NOVA5_DISABLED_PAIRS``).

    Returns:
        ``True`` if the pair should be excluded from collision checking.
    """
    for a, b in disabled_pairs:
        if (a == link1 and b == link2) or (a == link2 and b == link1):
            return True
    return False
