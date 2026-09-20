"""Evaluate a checkpoint using the same fixed-geometry model, on the remote host."""
from pathlib import Path
import os,runpy,sys
if __name__=='__main__':
    cloud=Path(__file__).resolve().parent/'baseten'
    os.chdir(cloud);sys.path.insert(0,str(cloud))
    sys.argv=[sys.argv[0],'--mode','evaluate',*sys.argv[1:]]
    runpy.run_path(str(cloud/'train.py'),run_name='__main__')
