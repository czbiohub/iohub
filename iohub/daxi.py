from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import dask
import dask.array as da
import numpy as np
import yaml

if TYPE_CHECKING:
    from _typeshed import StrOrBytesPath


class DaXiFOV:
    """
    Reader class for DaXi
    https://www.nature.com/articles/s41592-022-01417-2

    It provides an array-like API for DaXi timelapses
    that loads the volumes lazily.

    It assumes the channels and volumes have the same shape,
    when that is not the case the minimum value from each dimension is used.

    Parameters
    ----------
    data_path : StrOrBytesPath
        DaXi dataset path.
    missing_value : Optional[int], optional
        If provided this class won't raise an error when missing a volume and
        it will return an array with the provided value.
    """

    _CHANNELS_KEY = "Laser wavelengths"
    _SHAPE_KEY = "Dataset dimension"
    _SHAPE_IDX = ("T", "V", "Z", "Y", "X")

    def __init__(
        self,
        data_path: "StrOrBytesPath",
        missing_value: Optional[int] = None,
    ):
        super().__init__()

        self._data_path = Path(data_path)

        with open(self._data_path / "metadata.yaml") as f:
            self._metadata = yaml.safe_load(f)

        self._dtype = np.uint16

        if missing_value is None:
            self._missing_value = np.iinfo(self._dtype).max
        else:
            self._missing_value = missing_value

        self._wavelengths = self._metadata[self._CHANNELS_KEY]

        shape_dict = self._metadata[self._SHAPE_KEY]

        self._channels = [
            f"v{v}_c{wl}"
            for v in range(shape_dict["V"])
            for wl in self._wavelengths
        ]

        self._raw_shape = tuple(shape_dict[k] for k in self._SHAPE_IDX)

        shape_dict["Z"] //= len(self._wavelengths)
        shape_dict["V"] *= len(self._wavelengths)
        # y and x are flipped in comparison to DaXi file format
        shape_dict["Y"], shape_dict["X"] = shape_dict["X"], shape_dict["Y"]

        self._shape = tuple(shape_dict[k] for k in self._SHAPE_IDX)

        self._data = da.stack(  # T
            [
                da.concatenate(  # V
                    [
                        da.stack(  # C
                            [
                                da.from_delayed(
                                    self._load_volume(t, v, c),
                                    shape=self._shape[2:],
                                    dtype=self._dtype,
                                )
                                for c in range(len(self._wavelengths))
                            ]
                        )
                        for v in range(self._raw_shape[1])
                    ]
                )
                for t in range(self._shape[0])
            ]
        )  # T, V * C, Y, X shape

    @dask.delayed
    def _load_volume(self, t: int, v: int, c: int) -> np.ndarray:
        """
        Load a volume from disk.
        If the volume is missing it returns an array with the missing value.

        Parameters
        ----------
        t : int
            time index.
        v : int
            view index.
        c : int
            channel index.

        Returns
        -------
        np.ndarray
            Volume array.
        """
        path = self._volume_path(t, v)

        if not path.exists():
            return np.full(
                self._shape[2:], self._missing_value, dtype=self._dtype
            )

        arr = np.memmap(
            self._volume_path(t, v),
            dtype=self._dtype,
            shape=self._raw_shape[2:],
            mode="r",
        )[c :: len(self._wavelengths)]
        # inverting y and x and flipping new x to match original DaXi format
        return np.flip(arr.transpose((0, 2, 1)), axis=-1)

    def _volume_path(self, t: int, v: int) -> Path:
        """
        Return the path for a volume.

        Parameters
        ----------
        t : int
            time index.
        v : int
            view index.

        Returns
        -------
        Path
            Volume path.
        """
        z, y, x = self._raw_shape[2:]
        volume_path = (
            self._data_path / f"T_{int(t)}.V_{int(v)}.({z}x{y}x{x}).raw"
        )
        return volume_path

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Timelapse shape
        """
        return self._shape

    @property
    def channels(self) -> list[str]:
        """Return sorted channels name."""
        return self._channels

    def __getitem__(self, index) -> da.Array:
        """Lazily load array as indexed.

        Parameters
        ----------
        index : Array index.
            An indexing key as in numpy, but a bit more limited.

        Returns
        -------
        np.ndarray
            Output array.
        """
        return self._data[index]

    def __setitem__(self, key: Any, value: Any) -> None:
        raise PermissionError("DaXiFOV is read-only.")

    @property
    def ndim(self) -> int:
        return 5

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def metadata(self) -> dict[str, Any]:
        """Summarizes Clear Control metadata into a dictionary."""
        return self._metadata

    @property
    def scale(self) -> list[float]:
        """Dataset temporal, channel and spacial scales."""
        # TODO: use actual metadata, information is missing from file format
        return [
            1.0,
            1.0,
            1.24,
            0.439,
            0.439,
        ]

    @staticmethod
    def is_valid_path(path: "StrOrBytesPath") -> bool:
        """Check if a path is a valid DaXi dataset."""
        path = Path(path)
        return path.exists() and (path / "metadata.yaml").exists()


def create_mock_daxi_dataset(path: "StrOrBytesPath") -> None:
    """
    Creates a (2, 4, 64, 64, 64) Clear Control dataset of random integers.

    Parameters
    ----------
    path : StrOrBytesPath
        Dataset output path.
    """
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)

    rng = np.random.default_rng(0)
    array = rng.integers(
        0, 1_500, size=(2, 2, 2 * 64, 64, 64), dtype=np.uint16
    )

    metadata = {
        "Laser wavelengths": ["488", "561"],
        "Dataset dimension": {
            "T": 2,
            "V": 2,
            "X": 64,
            "Y": 64,
            "Z": 2 * 64,
        },
    }
    with open(path / "metadata.yaml", "w+") as f:
        yaml.dump(metadata, f)

    for t in range(array.shape[0]):
        for v in range(array.shape[1]):
            out_path = path / f"T_{t}.V_{v}.(128x64x64).raw"
            raw_map = np.memmap(
                out_path, dtype=np.uint16, mode="w+", shape=array.shape[2:]
            )
            raw_map[...] = array[t, v]
            raw_map.flush()
