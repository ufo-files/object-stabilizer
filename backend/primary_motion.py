"""Primary image registration, independent of bright-core photometry."""
import cv2
import numpy as np


def displacement(template,image):
    if min(float(np.ptp(template)),float(np.ptp(image)))<.005:
        raise ValueError('The primary reference patch has insufficient contrast.')
    if np.array_equal(template,image):return np.zeros(2),1.
    try:
        score,warp=cv2.findTransformECC(template,image,np.eye(2,3,dtype=np.float32),
            cv2.MOTION_TRANSLATION,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-6),None,3)
    except cv2.error as error:
        raise ValueError('Primary image registration failed; no stationary result can be certified.') from error
    delta=warp[:,2].astype(float)
    if not np.isfinite(delta).all() or np.linalg.norm(delta)>20 or score<.6:
        raise ValueError('Primary image registration lost agreement with the selected object.')
    return delta,float(score)


class PrimaryImageMotion:
    def __init__(self):
        self.first=None;self.previous=None;self.origins=[];self.sequential=[]
        self.absolute=[];self.absolute_scores=[];self.scores=[]

    def add(self,frame,point):
        center=np.rint(point).astype(int);origin=center-32
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray/=65535 if frame.dtype==np.uint16 else 255
        image=cv2.getRectSubPix(gray,(65,65),tuple(map(float,center)))
        if self.first is None:
            # Check observability even when comparing a frame to itself.
            displacement(image,image)
            self.first=image;delta=np.zeros(2);score=1.;absolute=np.zeros(2);absolute_score=1.
        else:
            shift,score=displacement(self.previous,image)
            delta=self.sequential[-1]+origin-self.origins[-1]+shift
            try:
                shift,absolute_score=displacement(self.first,image)
                absolute=origin-self.origins[0]+shift
            except ValueError:
                absolute=delta.copy();absolute_score=0.
        self.origins.append(origin);self.sequential.append(delta);self.scores.append(score)
        self.absolute.append(absolute);self.absolute_scores.append(absolute_score);self.previous=image

    def positions(self,times,anchor):
        times=np.asarray(times,float);sequential=np.asarray(self.sequential);absolute=np.asarray(self.absolute)
        if len(times)!=len(sequential) or not len(times):raise ValueError('Primary image/frame count mismatch.')
        if not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError('Primary frame times must increase.')
        if len(times)==1:return np.asarray(anchor)[None,:],np.asarray(self.scores)
        scores=np.asarray(self.absolute_scores)
        anchor_weights=.5*np.where(scores>=.85,scores**4,0.)
        pair_weights=np.maximum(.1,np.asarray(self.scores)[1:]**4)
        # Joint least squares of relative image motion and initial-image matches.
        # Solve the tridiagonal normal equations in O(frames), anchoring frame 0.
        # Known camera jumps satisfy both sets of observations and are preserved.
        diagonal=anchor_weights.copy();diagonal[:-1]+=pair_weights;diagonal[1:]+=pair_weights
        rhs=anchor_weights[:,None]*absolute
        increments=sequential[1:]-sequential[:-1]
        rhs[1:]+=pair_weights[:,None]*increments;rhs[:-1]-=pair_weights[:,None]*increments
        diagonal=diagonal[1:].copy();rhs=rhs[1:].copy();off=-pair_weights[1:]
        for i in range(1,len(diagonal)):
            multiplier=off[i-1]/diagonal[i-1]
            diagonal[i]-=multiplier*off[i-1];rhs[i]-=multiplier*rhs[i-1]
        corrected=np.zeros_like(sequential);corrected[-1]=rhs[-1]/diagonal[-1]
        for i in range(len(diagonal)-2,-1,-1):
            corrected[i+1]=(rhs[i]-off[i]*corrected[i+2])/diagonal[i]
        return np.asarray(anchor)+corrected,np.asarray(self.scores)
