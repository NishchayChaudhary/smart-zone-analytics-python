"""Tracking module for Smart Zone Analytics.

Exposes the Tracker class used to associate detections across frames
into persistent object tracks (IDs).
"""

from src.tracking.tracker import Tracker, TrackedObject

__all__ = ["Tracker", "TrackedObject"]