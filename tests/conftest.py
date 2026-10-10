"""Pytest defaults that do not open a GUI."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
