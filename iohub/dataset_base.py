from abc import ABC, abstractmethod
from typing import Any, Generator

import numpy as np
from numpy.typing import ArrayLike


class BaseFOV(ABC):
    # NOTE: not using the `data` method from ngff Position

    @property
    @abstractmethod
    def channels(self) -> list[str]:
        raise NotImplementedError

    def channel_index(self, key: str) -> int:
        """Return index of given channel."""
        return self.channels.index(key)

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int, int, int, int]:
        # NOTE: suggestion, fix dimension to 5?
        # We could me more restrictive than ome-zarr
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, key: Any) -> ArrayLike:
        """
        Output object must support __array__ interface, np.asarray(...).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def ndim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        raise NotImplementedError

    @property
    def scale(self) -> list[float]:
        """Helper function for FOV scale."""
        raise NotImplementedError


class BaseFOVCollection(ABC):
    @abstractmethod
    def __getitem__(self, key: str) -> BaseFOV:
        """FOV key position to FOV object."""
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Generator[tuple[str, BaseFOV], None, None]:
        # NOTE is this preferred than the current `.positions` ?
        raise NotImplementedError