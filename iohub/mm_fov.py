from __future__ import annotations

from pathlib import Path


from iohub.fov import BaseFOV, BaseFOVMapping


class MicroManagerFOV(BaseFOV):
    def __init__(self, parent: MicroManagerFOVMapping, key: int) -> None:
        self._position = key
        self._parent = parent

    @property
    def parent(self) -> MicroManagerFOVMapping:
        return self._parent

    @property
    def root(self) -> Path:
        return self.parent.root

    @property
    def zyx_scale(self) -> tuple[float, float, float]:
        return self.parent.zyx_scale

    @property
    def channel_names(self) -> list[str]:
        return self.parent.channel_names

    def frame_metadata(self, t: int, z: int, c: int) -> dict | None:
        """
        Return image plane metadata for a given camera frame.

        Parameters
        ----------
        t : int
            Time index.
        z : int
            Z slice index.
        c : int
            Channel index.

        Returns
        -------
        dict | None
            Image plane metadata. None if not available.
        """
        raise NotImplementedError


class MicroManagerFOVMapping(BaseFOVMapping):
    def __init__(self):
        self._mm_meta: dict = None
        self._stage_positions: list[dict[str, str | float]] = []
        self.channel_names: list[str] = None

    @property
    def mm_meta(self):
        return self._mm_meta

    @mm_meta.setter
    def mm_meta(self, value):
        if not isinstance(value, dict):
            raise TypeError(
                f"Type of `mm_meta` should be `dict`, got `{type(value)}`."
            )
        self._mm_meta = value

    @property
    def stage_positions(self):
        return self._stage_positions

    @stage_positions.setter
    def stage_positions(self, value):
        if not isinstance(value, list):
            raise TypeError(
                f"Type of `stage_position` should be `list`, got `{type(value)}`."
            )
        self._stage_positions = value

    @property
    def hcs_position_labels(self):
        """Parse plate position labels generated by the HCS position generator,
        e.g. 'A1-Site_0' or '1-Pos000_000', and split into row, column, and
        FOV names.

        Returns
        -------
        list[tuple[str, str, str]]
            FOV name paths, e.g. ('A', '1', '0') or ('0', '1', '000000')
        """
        if not self.stage_positions:
            raise ValueError("Stage position metadata not available.")
        try:
            # Look for "'A1-Site_0', 'H12-Site_1', ... " format
            labels = [
                pos["Label"].split("-Site_") for pos in self.stage_positions
            ]
            return [(well[0], well[1:], fov) for well, fov in labels]
        except Exception:
            try:
                # Look for "'1-Pos000_000', '2-Pos000_001', ... "
                # and split into ('1', '000_000'), ...
                labels = [
                    pos["Label"].split("-Pos") for pos in self.stage_positions
                ]
                # remove underscore from FOV name, i.e. '000_000'
                # collect all wells in row '0' so output is
                # ('0', '1', '000000')
                return [
                    ("0", col, fov.replace("_", "")) for col, fov in labels
                ]
            except Exception:
                labels = [pos.get("Label") for pos in self.stage_positions]
                raise ValueError(
                    "HCS position labels are in the format of "
                    "'A1-Site_0', 'H12-Site_1', or '1-Pos000_000' "
                    f"Got labels {labels}"
                )
