import pytest
from utils import read_data
import numpy as np


@pytest.fixture
def test_data_path():
    return "/path/to/test/data"


def test_read_data_returns_empty_dict_when_path_does_not_exist(test_data_path, mocker):
    mocker.patch("os.path.join", return_value=test_data_path)
    mocker.patch("os.listdir", side_effect=FileNotFoundError)

    result = read_data(test_data_path)

    assert result == {}
