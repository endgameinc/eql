"""Directory for miscellaneous files."""
import os

DIR = os.path.dirname(os.path.abspath(__file__))
ETC_DIR = os.path.dirname(os.path.abspath(__file__))


def get_etc_path(*relative_path):
    """Get the absolute path for an etc file."""
    return os.path.abspath(os.path.join(ETC_DIR, *relative_path))


def get_etc_file(*relative_path):
    """Get a file from the etc directory."""
    path = get_etc_path(*relative_path)
    with open(path, 'r') as f:
        return f.read()
