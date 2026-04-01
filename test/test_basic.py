import os

def test_src_folder_exists():
    assert os.path.isdir("src")

def test_requirements_exists():
    assert os.path.isfile("requirements.txt")