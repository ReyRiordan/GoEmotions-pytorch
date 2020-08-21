import pytest
from pathlib import Path


@pytest.fixture(scope='function',autouse=True)
def test_log(request) -> None:
    print()
    print(f"========== {request.node.nodeid} START ========")

    def fin():
        print()
        print(f"========== {request.node.nodeid} END ==========")
    request.addfinalizer(fin)


@pytest.fixture
def project_root_dir_path() -> Path:
    """
    Get the project root directory as a Path object.

    Returns
    -------
    project root directory

    """

    # root = ../..
    return Path(__file__).parent.parent


@pytest.fixture
def test_fixtures_dir_path() -> Path:
    """
    Get {project_root_dir_path}/test_fixtures as a Path object

    Returns
    -------
    Path

    """
    return Path(__file__).parent.parent / 'test_fixtures'
