"""Repository detector plus fixed-marker camera pose for an arm-mounted camera."""
import json
from pathlib import Path
import threading
import time
import cv2
import numpy as np
from vision.pancake_detection import (PancakeDetector, open_source, resize_for_display,
                                     estimate_hsv_range, draw_detection)
from core import ray_plane,rotation,pan_pose


class Capture:
    """One newest frame only. Receipt time is not an exposure timestamp."""
    def __init__(self,source):
        self.cap,self.live,self.description=open_source(source)
        if self.cap is None:raise RuntimeError('Camera could not be opened')
        self.latest=None;self.error=None;self.lock=threading.Lock();self.stop=threading.Event()
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def loop(self):
        seq=0
        try:
            while not self.stop.is_set():
                exposed=None
                if hasattr(self.cap,'queue'):
                    # OAK synchronized host clock: convert measured age to monotonic.
                    import depthai as dai
                    message=self.cap.queue.get()
                    age=(dai.Clock.now()-message.getTimestamp()).total_seconds()
                    received=time.monotonic();exposed=received-age
                    frame=message.getCvFrame();ok=True
                else:
                    ok,frame=self.cap.read();received=time.monotonic()
                if not ok or frame is None:raise RuntimeError('Camera feed ended or disconnected')
                seq+=1
                with self.lock:self.latest=(seq,received,exposed,resize_for_display(frame))
        except Exception as exc:self.error=str(exc)
    def get(self):
        with self.lock:return self.latest
    def close(self):
        self.stop.set();self.cap.release();self.thread.join(timeout=1)


class Projector:
    def __init__(self,cfg,folder):
        self.cfg=cfg;self.folder=Path(folder);self.intrinsics=None;self.mount=None
        for name,attr in [('intrinsics_file','intrinsics'),('mount_file','mount')]:
            path=self.folder/cfg['calibration'][name]
            if path.exists():setattr(self,attr,json.loads(path.read_text()))
        dictionary=cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco,cfg['marker']['dictionary']))
        self.detector=cv2.aruco.ArucoDetector(dictionary)
        s=cfg['marker']['side_m']/2
        self.object_points=np.array([[-s,s,0],[s,s,0],[s,-s,0],[-s,-s,0]],np.float64)
        self.last_pose=None
    def pose(self,frame):
        if self.intrinsics is None:raise ValueError('Calibrate intrinsics first')
        if list(frame.shape[1::-1])!=self.intrinsics['image_size']:
            raise ValueError('Frame dimensions differ from camera calibration')
        K=np.array(self.intrinsics['K'],np.float64);D=np.array(self.intrinsics['D'],np.float64)
        corners,ids,_=self.detector.detectMarkers(frame)
        if ids is None:raise ValueError('Fixed reference marker not visible')
        ids=list(ids.ravel())
        if self.cfg['marker']['id'] not in ids:raise ValueError('Reference marker ID not visible')
        image_points=corners[ids.index(self.cfg['marker']['id'])].reshape(4,2).astype(np.float64)
        result=cv2.solvePnPGeneric(self.object_points,image_points,K,D,flags=cv2.SOLVEPNP_IPPE_SQUARE)
        options=[]
        for rv,tv in zip(result[1],result[2]):
            R=cv2.Rodrigues(rv)[0];t=tv.ravel();camera=-R.T@t
            if camera[2]<=0 or np.any((R@self.object_points.T+t[:,None])[2]<=0):continue
            projected=cv2.projectPoints(self.object_points,rv,tv,K,D)[0].reshape(4,2)
            error=float(np.sqrt(np.mean(np.sum((projected-image_points)**2,axis=1))))
            options.append((error,R,t,camera))
        if not options:raise ValueError('No physically valid marker pose')
        options.sort(key=lambda p:p[0]);error,R,t,camera=options[0]
        if error>self.cfg['tracking']['max_marker_reprojection_px']:raise ValueError('Marker reprojection error too large')
        if len(options)>1 and options[1][0]-error<.15:
            difference=np.linalg.norm(cv2.Rodrigues(R@options[1][1].T)[0])
            if difference>np.deg2rad(5):raise ValueError('Ambiguous marker pose; change view')
        self.last_pose={'Rcw':R.tolist(),'tcw':t.tolist(),'camera_world_m':camera.tolist(),
                        'marker_reprojection_px':error}
        return K,D,R,t,camera
    def world_observation(self,frame,detection):
        K,D,R,t,camera=self.pose(frame)
        if self.mount is None:raise ValueError('Save camera mount at physical zero with M')
        g=self.cfg['geometry']
        if g['joint_world_m'] is None or g['pan_center_from_joint_m'] is None:
            raise ValueError('Measure joint_world_m and pan_center_from_joint_m')
        R0=np.asarray(self.mount['Rcw']);camera0=np.asarray(self.mount['camera_world_m'])
        relative=R.T@R0
        angle=float(np.rad2deg(np.arctan2(relative[2,0],relative[0,0])))
        if np.linalg.norm(relative-rotation(angle))>.1:raise ValueError('Camera motion is not a rigid single-axis rotation')
        joint=np.asarray(g['joint_world_m'])
        neutral_pan=joint+np.asarray(g['pan_center_from_joint_m'])
        if abs(np.linalg.norm(camera0-neutral_pan)-g['camera_to_pan_center_m'])>.03:
            raise ValueError('Calibrated camera-to-pan distance differs from measured 14 inches')
        expected=joint+rotation(angle)@(camera0-joint)
        if np.linalg.norm(camera-expected)>self.cfg['tracking']['max_rigid_translation_error_m']:
            raise ValueError('Camera pose disagrees with measured joint geometry')
        if detection is None:raise ValueError('Pancake not detected')
        pixel=cv2.undistortPoints(np.array([[[detection['x'],detection['y']]]],np.float64),K,D,P=K).ravel()
        world=ray_plane(pixel,K,R,t,joint[1])
        center,normal=pan_pose(angle,g)
        height=float((world-center)@normal)
        return {'world_m':world.tolist(),'angle_deg':angle,'height_m':height,
                'marker_reprojection_px':self.last_pose['marker_reprojection_px']}
    def save_mount(self,frame):
        self.pose(frame)
        path=self.folder/self.cfg['calibration']['mount_file']
        path.write_text(json.dumps(self.last_pose,indent=2)+'\n');self.mount=self.last_pose.copy()
