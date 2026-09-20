#!/usr/bin/env python3
"""Calibrate the exact 640-wide workshop image using a measured chessboard."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from vision.pancake_detection import open_source,resize_for_display


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='0');p.add_argument('--columns',type=int,default=9)
    p.add_argument('--rows',type=int,default=6);p.add_argument('--square-mm',type=float,required=True)
    p.add_argument('--output',type=Path,default=Path(__file__).with_name('intrinsics.json'))
    a=p.parse_args()
    if a.square_mm<=0 or min(a.columns,a.rows)<3:p.error('Use positive square size and at least 3x3 inner corners')
    cap,_,_=open_source(a.source)
    if cap is None:raise RuntimeError('Camera unavailable')
    obj=np.zeros((a.columns*a.rows,3),np.float32)
    obj[:,:2]=np.mgrid[0:a.columns,0:a.rows].T.reshape(-1,2)*a.square_mm/1000
    objects=[];images=[];size=None
    print('SPACE saves a board view. Collect >=15 varied tilts/positions; C calibrates; Q quits.')
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            frame=resize_for_display(frame);size=frame.shape[1::-1]
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            found,corners=cv2.findChessboardCornersSB(gray,(a.columns,a.rows))
            if found:cv2.drawChessboardCorners(frame,(a.columns,a.rows),corners,found)
            cv2.putText(frame,f'{len(images)} samples; SPACE save / C fit',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),2)
            cv2.imshow('Calibration',frame);k=cv2.waitKey(1)&255
            if k in (27,ord('q')):break
            if k==32 and found:objects.append(obj.copy());images.append(corners.copy())
            if k==ord('c'):
                if len(images)<15:print('Need at least 15 varied views');continue
                error,K,D,_,_=cv2.calibrateCamera(objects,images,size,None,None)
                if not np.isfinite(error) or error>1:print(f'RMS {error:.2f}px too large; recollect images');continue
                a.output.write_text(json.dumps({'K':K.tolist(),'D':D.ravel().tolist(),'image_size':list(size),'rms_px':error},indent=2)+'\n')
                print(f'Saved {a.output}, RMS {error:.3f}px. Verify against held-out board views.');break
    finally:cap.release();cv2.destroyAllWindows()

if __name__=='__main__':main()
