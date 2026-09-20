#!/usr/bin/env python3
"""Independent A-only flip workshop. Defaults to camera-only; --live enables keys."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'robot'/'motor_control'))
import cv2
import numpy as np
from core import *
from tracking import Capture,Projector,PancakeDetector,estimate_hsv_range,draw_detection
from hardware import Shared,Recorder,Worker,validate,sequence_targets
import tuning
from keys import decode_key

HERE=Path(__file__).resolve().parent
WINDOW='Flip Workshop'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=HERE/'config.json')
    ap.add_argument('--source',help='0, 1, oak, auto, or video path (watch mode only)')
    ap.add_argument('--live',action='store_true');ap.add_argument('--port')
    ap.add_argument('--make-marker',type=Path,help='Write printable marker PNG and exit')
    args=ap.parse_args();cfg=validate(json.loads(args.config.read_text()))
    if args.make_marker:
        d=cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco,cfg['marker']['dictionary']))
        marker=cv2.aruco.generateImageMarker(d,cfg['marker']['id'],600)
        image=cv2.copyMakeBorder(marker,100,100,100,100,cv2.BORDER_CONSTANT,value=255)
        if not cv2.imwrite(str(args.make_marker),image):raise RuntimeError('Could not write marker')
        print(f"Print the BLACK square at {cfg['marker']['side_m']*1000:g} mm; preserve white border.");return
    if args.live and not args.port:ap.error('--live requires --port')
    source=args.source or cfg['source']
    shared=Shared();camera=None;worker=None;writer=None;recorder=None
    state={'frame':None};lastseq=None;frames=0;tune_open=False
    folder=ROOT/'data'/'logs'/'flip_workshop'/str(time.time_ns())
    def save():args.config.write_text(json.dumps(cfg,indent=2)+'\n')
    def click(event,x,y,flags,param):
        if shared.busy or state['frame'] is None:return
        frame=state['frame'];x-=max(0,(960-frame.shape[1])//2)
        if not (0<=x<frame.shape[1] and 0<=y<frame.shape[0]):return
        if event==cv2.EVENT_LBUTTONDOWN:
            color=estimate_hsv_range(cv2.cvtColor(state['frame'],cv2.COLOR_BGR2HSV),x,y)
            if color:cfg['vision']['hsv'].update(color);detector.reset();save()
        elif event==cv2.EVENT_RBUTTONDOWN:
            cfg['vision']['edge_target_px']=[x,y];save()
        elif event==cv2.EVENT_MBUTTONDOWN or (event==cv2.EVENT_LBUTTONDBLCLK):
            cfg['vision']['pan_center_px']=[x,y];save()
    try:
        recorder=Recorder(folder);recorder.event('session',config=cfg,live=args.live)
        camera=Capture(source)
        if args.live and not camera.live:raise ValueError('Recorded video cannot command a live robot')
        detector=PancakeDetector();projector=Projector(cfg,args.config.parent)
        cv2.namedWindow(WINDOW);cv2.setMouseCallback(WINDOW,click)
        if args.live:worker=Worker(args.port,cfg,shared,recorder)
        print('Click pancake: color. Right-click: edge target. Double-click: resting pancake center.')
        print('Up/Down nudge | Z save zero | F/SPACE flip | T tune angles/times | S slide | C catch | N neutral | V calibrated vision flip')
        print('On first movement, excessive firmware acceleration is automatically lowered and saved; speed settings unchanged.')
        print('R reload config | 0 miss / 1 catch / 2 verified flip | X stop | Q quit. No automatic repeat.')
        while True:
            shared.heartbeat=time.monotonic()
            packet=camera.get()
            if camera.error:
                shared.message=camera.error
                if worker:worker.stop.set()
            if packet and packet[0]!=lastseq:
                seq,received,exposed,frame=packet;lastseq=seq;frames+=1;state['frame']=frame
                if exposed is None and cfg['tracking']['webcam_latency_s'] is not None:
                    exposed=received-cfg['tracking']['webcam_latency_s']
                if exposed is not None and (exposed>received or exposed<received-5):exposed=None
                detection,_,mask=detector.detect(frame,cfg['vision']['hsv'])
                obs={'seq':seq,'received':received,'exposed':exposed,'pixel':None,'score':0.,'trial':shared.trial}
                if detection:
                    obs.update(pixel=[detection['x'],detection['y']],score=detection['score'],
                               box=list(detection['box']),area=detection['area'])
                try:obs.update(projector.world_observation(frame,detection))
                except (ValueError,cv2.error) as exc:obs['projection_error']=str(exc)
                shared.observe(obs);recorder.event('frame',**obs,phase=shared.state,video_index=frames-1)
                if writer is None:
                    h,w=frame.shape[:2]
                    writer=cv2.VideoWriter(str(folder/'camera.avi'),cv2.VideoWriter_fourcc(*'MJPG'),30,(w,h))
                    if not writer.isOpened():raise RuntimeError('Video recorder could not open')
                writer.write(frame)
                view=frame.copy();draw_detection(view,detection)
                for key,color,radius in [('pan_center_px',(255,100,0),cfg['vision']['ready_radius_px']),('edge_target_px',(255,0,255),cfg['vision']['edge_radius_px'])]:
                    point=cfg['vision'][key]
                    if point is not None:cv2.circle(view,tuple(map(int,point)),int(radius),color,2)
                lines=[('LIVE ' if args.live else 'WATCH ')+shared.state,
                       shared.message,
                       (' > '.join(f'{q:g}' for q in [cfg.get('start_angle_deg',0)]+sequence_targets(cfg)) + ' | MAX FEED, NO DWELL') if cfg.get('three_part') else f"0 > {cfg['slide']['angle_deg']:g} > {cfg['launch']['angle_deg']:g} > {cfg['catch']['fallback_angle_deg']:g} > 0",
                       'Up/Down nudge | Z save zero | F/SPACE flip | T tune | V vision | X STOP | Q quit']
                cv2.imshow(WINDOW,tuning.status_panel(view,lines))
            raw_key=cv2.waitKeyEx(1)
            arrow=decode_key(raw_key)
            key=raw_key if 0<=raw_key<256 else 255
            if tune_open and tuning.handle_key(raw_key):key=255;arrow=None
            if key in (ord('q'),27):break
            if key==ord('x') and worker:worker.stop.set()
            if key in (ord('0'),ord('1'),ord('2')) and shared.trial and not shared.busy:
                recorder.event('operator_rating',trial=shared.trial,rating=chr(key),
                               meaning={'0':'miss','1':'caught_without_confirmed_flip','2':'opposite_side_down_confirmed'}[chr(key)])
                shared.message='Rating saved. Inspect/reset pancake; F or L starts another trial.'
            if shared.busy:continue
            if arrow:
                if worker:worker.submit(arrow,cfg)
                else:shared.message='Watch mode: motor nudges require --live --port'
                continue
            if key==ord('t'):
                if tune_open:cv2.destroyWindow(tuning.WINDOW);tune_open=False
                else:tuning.open_panel(cfg);tune_open=True
            if tune_open:
                if cv2.getWindowProperty(tuning.WINDOW,cv2.WND_PROP_VISIBLE)<1:tune_open=False
                else:
                    try:
                        new_cfg=tuning.read_panel(cfg)
                        if new_cfg!=cfg:
                            cfg=new_cfg;projector.cfg=cfg;save();shared.message='Angles/times saved; F to flip'
                    except ValueError as exc:shared.message=str(exc);continue
            if key==ord('r'):
                try:
                    new_cfg=validate(json.loads(args.config.read_text()));new_projector=Projector(new_cfg,args.config.parent)
                    cfg=new_cfg;projector=new_projector
                    detector.reset();shared.observations=[];shared.message='Configuration reloaded'
                    if tune_open:cv2.destroyWindow(tuning.WINDOW);tuning.open_panel(cfg)
                except Exception as exc:shared.message=f'Config error: {exc}'
            if key==ord('m') and state['frame'] is not None:
                try:
                    if args.live and not shared.reference:raise ValueError('Save physical neutral with Z first')
                    projector.save_mount(state['frame']);cfg['calibration']['verified']=False;save()
                    shared.message='Mount saved; verify measured geometry then set calibration.verified true and reload'
                except Exception as exc:shared.message=str(exc)
            actions={ord('z'):'zero',ord('p'):'start',ord('s'):'slide',ord('l'):'timed',ord('f'):'timed',32:'timed',ord('v'):'vision',ord('n'):'neutral',ord('c'):'catch_test'}
            if key in actions:
                if worker:worker.submit(actions[key],cfg)
                else:shared.message='Watch mode: launch with --live --port to enable motion keys'
            if cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1:break
    finally:
        if worker:worker.close()
        if camera:camera.close()
        if writer:writer.release()
        cv2.destroyAllWindows()
        if recorder:recorder.close()
        print(f'Logs and timestamped video index: {folder}')


if __name__=='__main__':main()
