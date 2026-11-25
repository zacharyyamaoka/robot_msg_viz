"""Sample data shipped with the package (the gallery's samples are built in code; these are the files they need).

- ``ur5e/``: the UR5e collision model from Universal Robots' ur_description (BSD-3-Clause, ``LICENSE.ur_description``
  beside it): the serial chain and its STL meshes, about 0.4 MB. The JointTrajectory sample is drawn on it.

WHY vendored here and not imported from robot_dashboards (which ships the same files): robot_msg_viz never imports
the packages built on it (docs/CONTRACT.md, "Layers").
"""

from pathlib import Path

ASSETS = Path(__file__).resolve().parent
UR5E_URDF = ASSETS / "ur5e" / "ur5e.urdf"
