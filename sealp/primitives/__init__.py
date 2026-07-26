"""
Motion Primitive Library
=========================

Reusable motion primitives for assembly operations.
Each primitive wraps WRS planners into a high-level interface.

Available primitives:
    - ``TransportPrimitive``         — single-arm pick-transport-place with mandatory Cartesian
      approach/depart segments
    - ``DirectTransportPrimitive``   — same motion and same single grasp, but the segments are
      connected by RRT in joint space instead of by forced straight lines
    - ``SingleArmRegraspPrimitive``  — single-arm transport that may put the object down and
      re-grasp it, for parts whose staging and goal orientations share no grasp
    - ``DualTransportPrimitive``     — dual-arm cooperative transport
"""

from .base import MotionPrimitive
from .transport import TransportPrimitive
from .direct_transport import DirectTransportPrimitive
from .regrasp import SingleArmRegraspPrimitive
from .dual_transport import DualTransportPrimitive
