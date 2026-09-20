"""Readable custom controls with signed numeric entry, not offset trackbars."""
import copy
import cv2
import numpy as np
from hardware import validate
WINDOW='Flip controls'
# label, group (None=root), key, display scale, min, max, +/- step
FIELDS=[('Slide angle (deg)','slide','angle_deg',1,None,None,1),
 ('Slide move (ms)','slide','move_s',1000,1,3000,10),
 ('Slide hold (ms)','slide','min_hold_s',1000,1,3000,10),
 ('Wiggle amplitude (deg)','slide','wiggle_deg',1,0,10,.25),
 ('Wiggle cycles','slide','wiggle_cycles',1,0,10,1),
 ('Wiggle move (ms)','slide','wiggle_move_s',1000,1,1000,10),
 ('Launch angle (deg)','launch','angle_deg',1,None,None,1),
 ('Launch move (ms)','launch','move_s',1000,1,3000,10),
 ('Catch angle (deg)','catch','fallback_angle_deg',1,None,None,1),
 ('Catch move (ms)','catch','fallback_move_s',1000,1,3000,10),
 ('Catch hold (ms)','catch','settle_s',1000,1,3000,10),
 ('Return move (ms)','return','move_s',1000,1,3000,10),
 ('Speed ceiling (deg/s)','limits','speed_deg_s',1,1,3000,25),
 ('Tempo multiplier',None,'tempo',1,.1,10,.25)]
_values=[];_bounds=[];_editing=None;_text='';_replace=True;_drag=None;_error=''

def cell(i):return (20+(i//7)*540,70+(i%7)*73)
def value_of(cfg,g,k):return (cfg[g] if g else cfg).get(k,0)
def text(img,s,xy,scale=.55,color=(230,230,230)):
    cv2.putText(img,str(s),xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)

def render(cfg):
    img=np.full((640,1100,3),28,np.uint8)
    text(img,'FLIP CONTROLS',(20,29),.8)
    text(img,'Click a number; type a signed value such as -8; Enter saves. Drag a bar or use +/- .',(20,54),.5)
    for i,(label,g,k,scale,lo,hi,step) in enumerate(FIELDS):
        x,y=cell(i);lo,hi=_bounds[i]
        text(img,label,(x,y+20),.52)
        cv2.rectangle(img,(x+265,y),(x+389,y+31),(75,75,75),-1)
        shown=_text if _editing==i else f'{_values[i]:g}'
        text(img,shown[-12:],(x+275,y+22),.57,(70,230,255) if _editing==i else (255,255,255))
        for off,symbol in ((400,'-'),(450,'+')):
            cv2.rectangle(img,(x+off,y),(x+off+38,y+31),(60,60,60),-1);text(img,symbol,(x+off+12,y+22),.65)
        cv2.line(img,(x,y+48),(x+488,y+48),(85,85,85),3)
        u=np.clip((_values[i]-lo)/(hi-lo),0,1)
        cv2.circle(img,(int(x+u*488),y+48),6,(70,220,255),-1)
        text(img,f'{lo:g}',(x,y+67),.38);text(img,f'{hi:g}',(x+455,y+67),.38)
    text(img,_error or ('Queued mode: angles + speed only; move times, holds, wiggle and tempo are unused.' if cfg.get('three_part') else 'F/Space flips from either window when not typing. X stops. T closes controls.'),(20,603),.49)
    text(img,f"Acceleration capped at {cfg['limits']['acceleration_deg_s2']:g} deg/s^2; faster requests stay within firmware limits.",(20,628),.48)
    return img

def _assign(i,v):
    global _error
    lo,hi=_bounds[i]
    if not np.isfinite(v) or not lo<=v<=hi:_error=f'Value must be between {lo:g} and {hi:g}';return False
    if FIELDS[i][2]=='wiggle_cycles' and v!=int(v):_error='Wiggle cycles must be a whole number';return False
    _values[i]=v;_error='';return True

def handle_key(key):
    global _editing,_text,_replace,_error
    if _editing is None:return False
    if key in (10,13):
        try:
            if _assign(_editing,float(_text)):_editing=None
        except ValueError:_error='Enter a number, e.g. -8 or 0.25'
    elif key==27:_editing=None;_error=''
    elif key in (8,127):_text='' if _replace else _text[:-1];_replace=False
    elif 0<=key<128 and chr(key) in '-+.0123456789':
        _text=chr(key) if _replace else _text+chr(key);_replace=False
    elif key==ord('x'):_editing=None;return False
    return True

def mouse(event,x,y,flags,param):
    global _editing,_text,_replace,_drag
    if event==cv2.EVENT_LBUTTONUP:_drag=None;return
    if event==cv2.EVENT_MOUSEMOVE and _drag is None:return
    if event not in (cv2.EVENT_LBUTTONDOWN,cv2.EVENT_MOUSEMOVE):return
    for i,field in enumerate(FIELDS):
        cx,cy=cell(i)
        if event==cv2.EVENT_LBUTTONDOWN and cy<=y<=cy+31:
            if cx+265<=x<=cx+389:_editing=i;_text=f'{_values[i]:g}';_replace=True;return
            for off,sign in ((400,-1),(450,1)):
                if cx+off<=x<=cx+off+38:
                    _editing=None;lo,hi=_bounds[i];_assign(i,float(np.clip(_values[i]+sign*field[6],lo,hi)));return
        if (event==cv2.EVENT_LBUTTONDOWN and cx<=x<=cx+488 and abs(y-cy-48)<12) or _drag==i:
            _drag=i;_editing=None;lo,hi=_bounds[i];v=lo+np.clip((x-cx)/488,0,1)*(hi-lo)
            v=round(v) if field[3]==1000 or field[2]=='wiggle_cycles' else round(v,2)
            _assign(i,v);return

def open_panel(cfg):
    global _values,_bounds,_editing,_error
    _values=[value_of(cfg,g,k)*scale for _,g,k,scale,_,_,_ in FIELDS]
    _bounds=[(cfg['limits']['angle_min_deg'],cfg['limits']['angle_max_deg']) if lo is None else (lo,hi) for _,_,_,_,lo,hi,_ in FIELDS]
    _editing=None;_error=''
    cv2.namedWindow(WINDOW,cv2.WINDOW_AUTOSIZE);cv2.setMouseCallback(WINDOW,mouse)
    cv2.imshow(WINDOW,render(cfg))

def read_panel(cfg):
    new=copy.deepcopy(cfg)
    for v,(_,g,k,scale,_,_,_) in zip(_values,FIELDS):(new[g] if g else new).__setitem__(k,v/scale)
    cv2.imshow(WINDOW,render(cfg))
    return validate(new)


def status_panel(view,lines):
    """Put readable wrapped status below the camera, never over its image."""
    width=max(960,view.shape[1]);wrapped=[]
    for line in lines:
        row=''
        for word in str(line).split():
            candidate=(row+' '+word).strip()
            if cv2.getTextSize(candidate,cv2.FONT_HERSHEY_SIMPLEX,.6,1)[0][0]>width-40 and row:
                wrapped.append(row);row=word
            else:row=candidate
        wrapped.append(row)
    result=np.full((view.shape[0]+35+28*len(wrapped),width,3),22,np.uint8)
    x=(width-view.shape[1])//2;result[:view.shape[0],x:x+view.shape[1]]=view
    for i,line in enumerate(wrapped):text(result,line,(20,view.shape[0]+28+i*28),.6)
    return result
