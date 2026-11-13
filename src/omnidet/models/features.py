from __future__ import annotations

from dataclasses import dataclass

import torch

# Canonical input strides for the OmniDet P2/P3/P4/P5 feature pyramid.
DEFAULT_PYRAMID_STRIDES: dict[str, int] = {"p2": 4, "p3": 8, "p4": 16, "p5": 32}
DEFAULT_PYRAMID_LEVELS: tuple[str, ...] = ("p2", "p3", "p4", "p5")


@dataclass(frozen=True)
class FeatureSpec:
    """Contract for the feature pyramid passed from the neck to the task heads.

    The backbone adapter and the neck normalize whatever a backbone produces
    into this contract, so a head binds to a ``FeatureSpec`` instead of to any
    particular backbone. ``channels`` is the uniform channel count the neck
    emits for every level.
    """

    levels: tuple[str, ...]
    strides: tuple[int, ...]
    channels: int

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError("FeatureSpec requires at least one level")
        if len(self.levels) != len(self.strides):
            raise ValueError(
                f"levels/strides length mismatch: {self.levels} vs {self.strides}"
            )
        if self.channels <= 0:
            raise ValueError(f"FeatureSpec channels must be positive, got {self.channels}")

    @classmethod
    def pyramid(
        cls, channels: int, levels: tuple[str, ...] = DEFAULT_PYRAMID_LEVELS
    ) -> "FeatureSpec":
        try:
            strides = tuple(DEFAULT_PYRAMID_STRIDES[level] for level in levels)
        except KeyError as exc:
            known = ", ".join(sorted(DEFAULT_PYRAMID_STRIDES))
            raise ValueError(f"unknown pyramid level {exc}; known levels: {known}") from exc
        return cls(levels=tuple(levels), strides=strides, channels=channels)

    @property
    def stride_by_level(self) -> dict[str, int]:
        return dict(zip(self.levels, self.strides, strict=True))

    @property
    def finest_level(self) -> str:
        """Highest-resolution level (the one with the smallest input stride)."""
        mapping = self.stride_by_level
        return min(mapping, key=mapping.get)

    def validate(self, features: dict[str, torch.Tensor]) -> None:
        """Raise ``ValueError`` if a feature dict does not satisfy this contract."""
        # Skip while tracing for export: shapes are static there, and the check
        # would otherwise emit a spurious TracerWarning into the export graph.
        if torch.jit.is_tracing():
            return
        missing = [level for level in self.levels if level not in features]
        if missing:
            raise ValueError(
                f"feature pyramid missing levels {missing}; got {sorted(features)}"
            )
        for level in self.levels:
            tensor = features[level]
            if tensor.dim() != 4:
                raise ValueError(
                    f"feature '{level}' must be a 4D NCHW tensor, got shape {tuple(tensor.shape)}"
                )
            if tensor.shape[1] != self.channels:
                raise ValueError(
                    f"feature '{level}' has {tensor.shape[1]} channels, expected {self.channels}"
                )
