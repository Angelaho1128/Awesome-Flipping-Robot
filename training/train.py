"""Training entry point for an already provisioned remote GPU workstation."""
from pathlib import Path
import os,runpy,sys
if __name__=='__main__':
    cloud=Path(__file__).resolve().parent/'baseten'
    os.chdir(cloud);sys.path.insert(0,str(cloud))
    runpy.run_path(str(cloud/'train.py'),run_name='__main__')
