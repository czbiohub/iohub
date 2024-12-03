from pathlib import Path

import numpy as np
import pytest

from iohub.clearcontrol import ArrayIndex
from iohub.daxi import DaXiFOV, create_mock_daxi_dataset


@pytest.fixture
def mock_daxi_dataset_path(tmp_path: Path) -> Path:
    ds_path = tmp_path / "daxi_dataset_raw"
    create_mock_daxi_dataset(path=ds_path)
    return ds_path


@pytest.mark.parametrize(
    "key",
    [
        1,
        (slice(None), 1),
        (0, [1, 2]),
        (-1, np.asarray([0, 3])),
        (slice(1), -2),
        (0, 1, 57, slice(32), slice(32, None)),
        (0, 0, slice(32)),
    ],
)
def test_DaXiFOV_indexing(
    mock_daxi_dataset_path: Path,
    key: ArrayIndex,
) -> None:
    fov = DaXiFOV(mock_daxi_dataset_path)

    # copy of whole array
    similar_arr = np.asarray(fov[...])

    # checking if indexing works as in numpy
    assert np.array_equal(similar_arr[key], np.asarray(fov[key]))
