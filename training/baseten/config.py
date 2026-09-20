from pathlib import Path
from runpy import run_path

# Truss loads this file by path without adding its directory to sys.path.
# Load the sibling explicitly so submission works from any working directory.
project = run_path(str(Path(__file__).resolve().with_name("cloud.py")))["project"]
# To resume, replace both None values with the prior job ID and the EXACT
# synced checkpoint name shown in the dashboard (one leaf checkpoint only).
RESUME_JOB_ID=None
RESUME_CHECKPOINT=None
training_project=project(mode="train",resume_job_id=RESUME_JOB_ID,resume_checkpoint=RESUME_CHECKPOINT)
