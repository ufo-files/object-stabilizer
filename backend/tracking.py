"""Local full-resolution tracking. Bright-point photometry and checked optical flow."""
from dataclasses import dataclass
import math
import cv2
import numpy as np

@dataclass
class Observation:
    x: float
    y: float
    confidence: float
    accepted: bool
    reason: str = ""
    def row(self):
        return dict(x=float(self.x), y=float(self.y), confidence=float(self.confidence), accepted=bool(self.accepted), reason=self.reason)

def grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

def crop(gray, point, radius):
    x,y=point; h,w=gray.shape
    x0=max(0,int(math.floor(x-radius)));y0=max(0,int(math.floor(y-radius)))
    x1=min(w,int(math.ceil(x+radius+1)));y1=min(h,int(math.ceil(y+radius+1)))
    return gray[y0:y1,x0:x1],x0,y0

class BrightPointTracker:
    def __init__(self, frame, point, size=24, radius=100):
        self.position=np.array(point,dtype=float);self.velocity=np.zeros(2);self.radius=radius
        self.size=max(8,min(128,size));self.lost=0;self.last_frame=grayscale(frame)
        candidates=self.detect(frame,self.position,self.size/2)
        if not candidates:raise ValueError("No distinct bright point at this selection. Click its center or use textured-object mode.")
        # Initial click identifies the component, not the brightest unrelated object.
        target=min(candidates,key=lambda c: np.linalg.norm(c[0]-self.position))
        self.position=target[0];self.flux=target[2]
        self.initial=self.position.copy()
        self.last_confidence=target[1]

    def detect(self, frame, center, radius):
        roi,x0,y0=crop(grayscale(frame),center,radius)
        if min(roi.shape,default=0)<5:return []
        data=roi.astype(np.float32)
        signal=cv2.GaussianBlur(data,(0,0),.8)-cv2.GaussianBlur(data,(0,0),4)
        noise=max(.35,float(np.median(np.abs(signal-np.median(signal)))*1.4826))
        threshold=max(2.5,noise*5)
        mask=(signal>threshold).astype(np.uint8)
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
        result=[]
        for label in range(1,count):
            area=int(stats[label,cv2.CC_STAT_AREA])
            if area<2 or area>max(300,self.size**2):continue
            weights=np.where(labels==label,np.maximum(signal-threshold*.5,0),0)
            flux=float(weights.sum())
            if flux<=0:continue
            ys,xs=np.nonzero(weights)
            values=weights[ys,xs]
            point=np.array([float(np.dot(xs,values)/flux+x0),float(np.dot(ys,values)/flux+y0)])
            peak=float(signal[labels==label].max());snr=peak/noise
            confidence=min(.999,max(0,1-math.exp(-(snr-4)/5)))
            result.append((point,confidence,flux,snr))
        return result

    def update(self, frame):
        predicted=self.position+self.velocity
        candidates=self.detect(frame,predicted,self.radius)
        candidates=[c for c in candidates if c[1]>=.55 and .08<=c[2]/self.flux<=12]
        if not candidates:
            self.lost+=1;self.velocity*=.5
            return Observation(*self.position,0,False,"No reliable bright point; position held")
        # Prefer temporal proximity; an unrelated brighter light must not steal lock.
        chosen=min(candidates,key=lambda c:np.linalg.norm(c[0]-predicted)+2*abs(math.log(c[2]/self.flux)))
        point,confidence,flux,_=chosen
        allowed=max(12,min(self.radius, max(self.size, np.linalg.norm(self.velocity)*3+8)))
        if np.linalg.norm(point-predicted)>allowed:
            self.lost+=1;self.velocity*=.5
            return Observation(*self.position,confidence,False,"Candidate outside motion gate; position held")
        delta=point-self.position
        self.velocity=.6*self.velocity+.4*delta
        self.position=point;self.flux=.95*self.flux+.05*flux;self.lost=0
        return Observation(*point,confidence,True)

class FlowTracker:
    """Track local features with forward/backward and anchor-patch checks."""
    def __init__(self,frame,point,size=48,radius=100):
        self.position=np.array(point,dtype=float);self.size=max(16,int(size));self.radius=radius
        self.previous=grayscale(frame);self.points=self.features(self.previous)
        self.initial=self.position.copy();self.template=cv2.getRectSubPix(self.previous,(self.size,self.size),tuple(self.position))
        self.reference=self.template.copy()
        if self.points is None or len(self.points)<3:raise ValueError("Not enough object features. Increase the tracking area or use bright-point mode.")
    def features(self,gray):
        mask=np.zeros(gray.shape,np.uint8)
        x,y=np.rint(self.position).astype(int);r=self.size//2
        mask[max(0,y-r):min(gray.shape[0],y+r+1),max(0,x-r):min(gray.shape[1],x+r+1)]=255
        return cv2.goodFeaturesToTrack(gray,40,.02,2,mask=mask,blockSize=3)
    def update(self,frame):
        gray=grayscale(frame)
        roi,x0,y0=crop(gray,self.position,self.radius+self.size/2)
        params=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
        proposal=None;flow_ok=False
        if self.points is not None and len(self.points)>=3:
            nxt,ok,_=cv2.calcOpticalFlowPyrLK(self.previous,gray,self.points,None,**params)
            if nxt is not None:
                back,reverse,_=cv2.calcOpticalFlowPyrLK(gray,self.previous,nxt,None,**params)
                if back is not None:
                    good=(ok.ravel()>0)&(reverse.ravel()>0)&(np.linalg.norm(back-self.points,axis=2).ravel()<.75)
                    if good.sum()>=3:
                        offsets=(nxt-self.points).reshape(-1,2)[good]
                        displacement=np.median(offsets,axis=0)
                        coherent=np.linalg.norm(offsets-displacement,axis=1)<1.5
                        if coherent.sum()>=3:proposal=self.position+np.median(offsets[coherent],axis=0);flow_ok=True
        if min(roi.shape)>=self.size:
            scores=cv2.matchTemplate(roi,self.reference,cv2.TM_CCOEFF_NORMED)
            _,score,_,loc=cv2.minMaxLoc(scores)
            template_point=np.array([x0+loc[0]+(self.size-1)/2,y0+loc[1]+(self.size-1)/2])
        else:score=0;template_point=self.position
        accepted=False
        if flow_ok and np.linalg.norm(proposal-self.position)<=self.radius:
            patch=cv2.getRectSubPix(gray,(self.size,self.size),tuple(proposal))
            agreement=float(cv2.matchTemplate(patch,self.reference,cv2.TM_CCOEFF_NORMED)[0,0])
            if agreement>.6:accepted=True;score=agreement
        if not accepted and score>.85 and np.linalg.norm(template_point-self.position)<max(12,self.size/2):
            proposal=template_point;accepted=True
        if accepted:self.position=proposal
        self.previous=gray;self.points=self.features(gray)
        return Observation(*self.position,float(np.clip(score,0,1)),accepted,"" if accepted else "Flow and appearance checks failed; position held")

def repair_gaps(rows,max_gap=.3):
    """Interpolate only short, bounded gaps bracketed by measured positions."""
    result=[dict(r) for r in rows];valid=[i for i,r in enumerate(rows) if r['accepted']]
    for a,b in zip(valid,valid[1:]):
        duration=rows[b]['time']-rows[a]['time']
        if b-a<=1 or duration>max_gap or math.hypot(rows[b]['x']-rows[a]['x'],rows[b]['y']-rows[a]['y'])>60:continue
        for j in range(a+1,b):
            t=(rows[j]['time']-rows[a]['time'])/duration
            result[j]['x']=rows[a]['x']+(rows[b]['x']-rows[a]['x'])*t
            result[j]['y']=rows[a]['y']+(rows[b]['y']-rows[a]['y'])*t
            result[j]['reconstructed']=True
    return result
