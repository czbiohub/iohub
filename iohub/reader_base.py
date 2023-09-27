import zarr
from typing import Union
from numpy.typing import DTypeLike, NDArray


class ReaderBase:
    def __init__(self):
        self.frames: int = None
        self.channels: int = None
        self.slices: int = None
        self.height: int = None
        self.width: int = None
        self.dtype: DTypeLike = None
        self._mm_meta: dict = None
        self._stage_positions: list[dict[str, Union[str, float]]] = []
        self.z_step_size: float = None
        self.channel_names: list[str] = None

    @property
    def shape(self):
        """Get the underlying data shape as a tuple.

        Returns
        -------
        tuple
            (frames, slices, channels, height, width)

        """
        return self.frames, self.channels, self.slices, self.height, self.width

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
                f"Type of `mm_meta` should be `dict`, got `{type(value)}`."
            )
        self._stage_positions = value

    def get_zarr(self, position: int) -> zarr.Array:
        """Get a zarr array for a given position.

        Parameters
        ----------
        position : int
            position (aka ome-tiff scene)

        Returns
        -------
        zarr.Array
        """
        raise NotImplementedError

    def get_array(self, position: int) -> NDArray:
        """Get a numpy array for a given position.

        Parameters
        ----------
        position : int
            position (aka ome-tiff scene)

        Returns
        -------
        NDArray
        """

    def get_image(self, p: int, t: int, c: int, z: int) -> NDArray:
        """Get the image slice at dimension P, T, C, Z.

        Parameters
        ----------
        p : int
            index of the position dimension
        t : int
            index of the time dimension
        c : int
            index of the channel dimension
        z : int
            index of the z dimension

        Returns
        -------
        NDArray
            2D image frame
        """
        raise NotImplementedError

    def get_num_positions(self) -> int:
        """Get total number of scenes referenced in ome-tiff metadata.

        Returns
        -------
        int
            number of positions
        """
        raise NotImplementedError

    @property
    def hcs_position_labels(self):
        """Parse plate position labels generated by the HCS position generator,
        e.g. 'A1-Site_0' or '1-Pos000_000', and split into row, column, and
        FOV names.

        Returns
        -------
        list[tuple[str, str, str]]
            FOV name paths, e.g. ('A', '1', '0') or ('0', '0', '1')
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
                # and split into ('1', 'Pos000_000'), ...
                labels = [
                    pos["Label"].split("-Pos") for pos in self.stage_positions
                ]
                # split '000_000' into ('000', '000')
                # and remove leading zeros so output is ('0', '0', '0')
                return [
                    (row, *[str(int(s)) for s in col_fov.split("_")])
                    for row, col_fov in labels
                ]
            except Exception:
                labels = [pos.get("Label") for pos in self.stage_positions]
                raise ValueError(
                    "HCS position labels are in the format of "
                    "'A1-Site_0', 'H12-Site_1', or '1-Pos000_000' "
                    f"Got labels {labels}"
                )
