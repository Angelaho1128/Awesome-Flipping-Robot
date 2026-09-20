from pathlib import Path
from runpy import run_path

# Truss loads this file by path without adding its directory to sys.path.
# Load the sibling explicitly so submission works from any working directory.
project = run_path(str(Path(__file__).resolve().with_name("cloud.py")))["project"]
training_project=project(mode="smoke")
