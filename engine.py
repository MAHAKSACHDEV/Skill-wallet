"""Offline multimodal classroom telemetry. No biometric identification or storage."""
from dataclasses import dataclass
import cv2
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from scipy.io import wavfile
from io import BytesIO

@dataclass
class Config:
    presence_threshold: float = .70
    window_seconds: float = 2.0
    match_distance: float = .16
    max_missing_seconds: float = 4.0
    low_engagement: float = 45.0
    low_duration_seconds: float = 300.0


def _detector():
    """Use bundled Haar cascade, avoiding any model download at runtime."""
    path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    detector = cv2.CascadeClassifier(path)
    if detector.empty():
        raise RuntimeError('Local OpenCV face detector is unavailable.')
    return detector


def process_video(path, config: Config, max_frames=15000, stride=3):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError('Unable to decode video. Try an MP4 encoded with H.264.')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0: fps = 25.0
    detector = _detector()
    tracks = {}
    rows = []
    frame_number = 0
    processed = 0
    try:
        while processed < max_frames:
            ok, frame = cap.read()
            if not ok: break
            frame_number += 1
            if frame_number % stride: continue
            processed += 1
            t = (frame_number - 1) / fps
            h, w = frame.shape[:2]
            if max(h, w) > 960:
                scale = 960 / max(h, w)
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
                h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes = detector.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(28,28))
            centers = [((x+bw/2)/w, (y+bh/2)/h, bw/w, bh/h) for x,y,bw,bh in boxes]
            available = set(tracks)
            visible = []
            for cx, cy, bw, bh in centers:
                candidates = [(np.hypot(cx-tracks[k]['center'][0], cy-tracks[k]['center'][1]),k)
                              for k in available if t-tracks[k]['last'] <= config.max_missing_seconds]
                distance, key = min(candidates, default=(float('inf'),None))
                if distance > config.match_distance:
                    key = f'P{len(tracks)+1:03d}'
                    tracks[key] = {'center':(cx,cy),'first':t,'last':t,'observed':0.0,'samples':0,'forward':0}
                else:
                    available.remove(key)
                tr = tracks[key]
                tr['center'] = (cx,cy)
                tr['last'] = t
                tr['observed'] += stride/fps
                tr['samples'] += 1
                # Frontal detector is only a weak proxy for facing camera, NOT attention.
                tr['forward'] += 1
                visible.append(key)
            rows.append({'time_s':t,'visible_count':len(visible),
                         'visual_proxy':min(1.0,len(visible)/max(1,len(tracks)))})
    finally:
        cap.release()
    duration = frame_number / fps
    register = []
    for key,tr in tracks.items():
        observed = min(duration,tr['observed'])
        ratio = observed/max(duration,1e-6)
        register.append({'Student ID':key,'First Seen (s)':round(tr['first'],1),
                         'Last Seen (s)':round(tr['last'],1),
                         'Total Duration (s)':round(observed,1),
                         'Presence %':round(100*ratio,1),
                         'Status':'Present' if ratio > config.presence_threshold else 'Absent'})
    return pd.DataFrame(register,columns=['Student ID','First Seen (s)','Last Seen (s)','Total Duration (s)','Presence %','Status']),pd.DataFrame(rows),duration,frame_number>=max_frames


def process_photo(data):
    array = np.frombuffer(data,dtype=np.uint8)
    frame = cv2.imdecode(array,cv2.IMREAD_COLOR)
    if frame is None: raise ValueError('Invalid image file.')
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    boxes = _detector().detectMultiScale(gray,1.12,5,minSize=(28,28))
    for x,y,w,h in boxes:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(36,170,130),2)
    return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),len(boxes)


def process_audio(data,window_seconds=2.0):
    """WAV-only deterministic acoustic heuristics; not speaker identification."""
    sr, samples = wavfile.read(BytesIO(data))
    if samples.ndim>1: samples=samples.astype(np.float64).mean(axis=1)
    if np.issubdtype(samples.dtype,np.integer):
        samples = samples.astype(np.float64)/max(abs(np.iinfo(samples.dtype).min),np.iinfo(samples.dtype).max)
    else: samples = samples.astype(np.float64)
    if sr < 8000 or sr > 192000: raise ValueError('Unsupported audio sample rate.')
    length = max(1,int(sr*window_seconds))
    rows=[]
    for start in range(0,len(samples),length):
        chunk=samples[start:start+length]
        if len(chunk)<max(1,int(sr*.2)): continue
        rms=float(np.sqrt(np.mean(chunk**2)))
        crossings=float(np.mean(np.diff(np.signbit(chunk)))) if len(chunk)>1 else 0
        power=np.abs(np.fft.rfft(chunk))**2
        freqs=np.fft.rfftfreq(len(chunk),1/sr)
        total=float(power.sum())+1e-12
        speech_band=float(power[(freqs>=100)&(freqs<=3500)].sum()/total)
        # These categories are approximate acoustic states, not reliable speaker roles.
        if rms<.008: label='Silence'
        elif rms>.18 or (crossings>.22 and rms>.035): label='High Ambient Distraction'
        elif speech_band>.72 and rms>.018: label='Speech / Verbal Interaction'
        else: label='Ambient / Uncertain'
        rows.append({'time_s':start/sr,'rms':rms,'speech_band':speech_band,
                     'zero_crossing':crossings,'audio_state':label,
                     'interaction_proxy':.65 if label=='Speech / Verbal Interaction' else (.2 if label=='Silence' else .3)})
    return pd.DataFrame(rows)


def fuse(visual,audio,window_seconds=2.0):
    if visual is None or visual.empty: return pd.DataFrame()
    v=visual.copy()
    v['bucket']=(v.time_s//window_seconds).astype(int)
    v=v.groupby('bucket',as_index=False).agg(time_s=('time_s','mean'),visual_proxy=('visual_proxy','mean'),visible_count=('visible_count','max'))
    if audio is not None and not audio.empty:
        a=audio.copy();a['bucket']=(a.time_s//window_seconds).astype(int)
        a=a.groupby('bucket',as_index=False).agg(interaction_proxy=('interaction_proxy','mean'))
        v=v.merge(a,on='bucket',how='left')
    else: v['interaction_proxy']=np.nan
    # Explicitly avoid inventing an audio signal if absent.
    has_audio=v.interaction_proxy.notna()
    v['engagement_index']=np.where(has_audio,100*(.7*v.visual_proxy+.3*v.interaction_proxy),100*v.visual_proxy)
    v['engagement_index']=v.engagement_index.clip(0,100).round(1)
    return v


def advisories(fused,config:Config):
    if fused is None or fused.empty:return ['Process a classroom video to generate evidence-based prompts.']
    low=fused.engagement_index < config.low_engagement
    run=0; longest=0
    for flag in low:
        run=run+1 if flag else 0
        longest=max(longest,run)
    prompts=[]
    if longest*config.window_seconds>=config.low_duration_seconds:
        prompts.append('Sustained low collective signal: try a quick poll, pair discussion, or a 2-minute stretch break.')
    if fused.engagement_index.mean()<55:
        prompts.append('Consider checking comprehension with an anonymous one-question quiz.')
    if not prompts:prompts.append('No sustained low-signal pattern detected. Continue teaching and check understanding directly.')
    return prompts
