"""Sensor-only Bayesian local mapping and calibrated projection, independent of ROS."""
import math
import numpy as np


def quaternion_matrix(q):
    x,y,z,w=q
    norm=x*x+y*y+z*z+w*w
    if norm<1e-12:raise ValueError("invalid quaternion")
    s=2/norm
    return np.array([[1-s*(y*y+z*z),s*(x*y-z*w),s*(x*z+y*w)],
                     [s*(x*y+z*w),1-s*(x*x+z*z),s*(y*z-x*w)],
                     [s*(x*z-y*w),s*(y*z+x*w),1-s*(x*x+y*y)]])


def transform_points(points,translation,quaternion):
    return np.asarray(points)@quaternion_matrix(quaternion).T+np.asarray(translation)


def project(points,k,width,height):
    points=np.asarray(points)
    safe=np.maximum(points[:,2],1e-6)
    u=k[0]*points[:,0]/safe+k[2]
    v=k[4]*points[:,1]/safe+k[5]
    valid=(points[:,2]>.05)&(u>=2)&(u<width-2)&(v>=2)&(v<height-2)
    return np.stack([u,v],axis=1),valid


def filter_scan(ranges):
    r=np.array(ranges,dtype=float)
    # Remove isolated gross returns, preserving thin objects supported by neighbours.
    left,right=np.roll(r,1),np.roll(r,-1)
    with np.errstate(invalid="ignore"):
        outlier=np.isfinite(r)&np.isfinite(left)&np.isfinite(right)&(
            np.abs(r-(left+right)/2)>.25)&(np.abs(left-right)<.10)
    r[outlier]=np.nan
    return r


class LocalGrid:
    def __init__(self,resolution=.04,size=12.0,ttl=4.0):
        self.resolution=resolution
        self.width=round(size/resolution)
        self.origin=np.array([-size/2,-size/2])
        self.ttl=ttl
        self.odds=np.zeros((self.width,self.width),dtype=np.float32)
        self.seen=np.full(self.odds.shape,-np.inf,dtype=float)
        self.rgb=np.zeros((*self.odds.shape,3),dtype=np.float32)
        self.rgb_seen=np.full(self.odds.shape,-np.inf,dtype=float)

    def recenter(self, xy):
        """Move a rolling grid in integer cells, preserving measured overlap."""
        center=self.origin+self.width*self.resolution/2
        delta=np.asarray(xy)-center
        if np.max(np.abs(delta))<self.width*self.resolution/4:return
        shift=np.rint(delta/self.resolution).astype(int)
        for name,fill in (("odds",0),("seen",-np.inf),("rgb",0),("rgb_seen",-np.inf)):
            old=getattr(self,name);new=np.full_like(old,fill)
            dx,dy=map(int,shift)
            if abs(dx)<self.width and abs(dy)<self.width:
                sx=slice(max(0,dx),min(self.width,self.width+dx))
                sy=slice(max(0,dy),min(self.width,self.width+dy))
                tx=slice(max(0,-dx),min(self.width,self.width-dx))
                ty=slice(max(0,-dy),min(self.width,self.width-dy))
                new[ty,tx]=old[sy,sx]
            setattr(self,name,new)
        self.origin+=shift*self.resolution

    def index(self,points):
        cells=np.floor((np.asarray(points)-self.origin)/self.resolution+1e-9).astype(int)
        valid=np.all((cells>=0)&(cells<self.width),axis=-1)
        return cells,valid

    def update(self,start,endpoints,hits,stamp):
        start=np.asarray(start,dtype=float)
        endpoints=np.asarray(endpoints,dtype=float)
        if not len(endpoints):return
        expired=(stamp-self.seen)>self.ttl
        self.odds[expired]=0
        vectors=endpoints-start
        steps=np.ceil(np.linalg.norm(vectors,axis=1)/self.resolution).astype(int)
        max_steps=int(max(steps,default=0))
        if max_steps<1:return
        # Endpoint is handled separately; never clear behind a measured return.
        t=np.arange(max_steps)[None,:]/np.maximum(steps[:,None],1)
        points=start+vectors[:,None,:]*t[:,:,None]
        cells,valid=self.index(points)
        valid &= (np.arange(max_steps)[None,:]<steps[:,None])
        free=np.unique((cells[:,:,1]*self.width+cells[:,:,0])[valid])
        end_cells,end_valid=self.index(endpoints)
        occupied=np.unique((end_cells[:,1]*self.width+end_cells[:,0])[end_valid&np.asarray(hits)])
        free=np.setdiff1d(free,occupied,assume_unique=True)
        flat=self.odds.ravel()
        flat[free]-=.40
        flat[occupied]+=.85
        np.clip(flat,-3.5,3.5,out=flat)
        self.seen.ravel()[np.union1d(free,occupied)]=stamp

    def occupancy(self,stamp):
        values=np.rint(100/(1+np.exp(-self.odds))).astype(np.int16)
        values[stamp-self.seen>self.ttl]=-1
        return values

    def colorize(self,points,colors,stamp):
        cells,valid=self.index(points)
        for (x,y),color in zip(cells[valid],np.asarray(colors)[valid]):
            self.rgb[y,x]=color
            self.rgb_seen[y,x]=stamp


class ClusterTracker:
    def __init__(self):
        self.tracks={}
        self.next_id=1

    def update(self,clusters,stamp):
        self.tracks={key:t for key,t in self.tracks.items() if stamp-t["stamp"]<1.5}
        used=set()
        for center in clusters:
            center=np.asarray(center)
            candidates=[(np.linalg.norm(center-t["xy"]),key) for key,t in self.tracks.items()
                        if key not in used and stamp>t["stamp"]]
            distance,key=min(candidates,default=(math.inf,None))
            if distance<.35:
                t=self.tracks[key]
                dt=stamp-t["stamp"]
                velocity=(center-t["xy"])/dt
                # Reject discontinuous association; centroid motion is not perfect object velocity.
                if np.linalg.norm(velocity)<1.2:
                    t["velocity"]=.6*t["velocity"]+.4*velocity
                    t["count"]+=1
                else:t["velocity"]*=0
                t.update(xy=center,stamp=stamp)
            else:
                key=self.next_id;self.next_id+=1
                self.tracks[key]={"xy":center,"velocity":np.zeros(2),"count":1,"stamp":stamp}
            used.add(key)
        return [{"id":key,"xy":t["xy"].tolist(),"velocity":t["velocity"].tolist(),
                 "moving":bool(t["count"]>=8 and np.linalg.norm(t["velocity"])>.10),
                 "age_s":stamp-t["stamp"]} for key,t in self.tracks.items()]


def occupancy_to_pgm(grid):
    """Preserve unknown/intermediate cells under free=.35, occupied=.65."""
    grid=np.asarray(grid)
    return np.where((grid<0)|((grid>35)&(grid<65)),128,
                    np.where(grid>=65,0,254)).astype(np.uint8)
