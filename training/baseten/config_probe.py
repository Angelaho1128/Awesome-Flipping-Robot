from pathlib import Path
from runpy import run_path
project=run_path(str(Path(__file__).resolve().with_name('cloud.py')))['project']
training_project=project(mode='probe')
