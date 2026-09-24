"""Optional visual aiming aids computed only from the observed RGB images."""
import numpy as np
from PIL import Image


def aim_marker(frame):
    marked=np.asarray(frame).copy()
    h,w=marked.shape[:2];x,y=w//2,h//2
    # Four small red arms, preserving the center pixels of the actual target.
    marked[y-1:y+2,x-8:x-3]=(255,35,35)
    marked[y-1:y+2,x+4:x+9]=(255,35,35)
    marked[y-8:y-3,x-1:x+2]=(255,35,35)
    marked[y+4:y+9,x-1:x+2]=(255,35,35)
    return marked


def aim_messages(messages, current, mode):
    if mode=='plain':return messages
    if mode in ('current','current-large'):
        # Diagnostic/policy variant: avoid confusing a removed block in the old
        # frame with the current target. Keep only the CURRENT original RGB.
        frame=current
        if mode=='current-large':
            h,w=current.shape[:2]
            frame=np.asarray(Image.fromarray(current).resize((w*3,h*3),Image.Resampling.NEAREST))
        output=[]
        for message in messages:
            if isinstance(message['content'],str):
                output.append(dict(message))
            else:
                parts=message['content']
                after=max(i for i,p in enumerate(parts) if p['type']=='image')+1
                output.append(dict(message,content=[
                    dict(type='text',text='Only the CURRENT scene is shown. Aim is exactly at image center.'),
                    dict(type='image',image=frame),*parts[after:]]))
        return output
    if mode not in ('aim','crop','crop-feedback-last','target-detail'):raise ValueError(mode)
    output=[]
    for message in messages:
        if isinstance(message['content'],str):
            output.append(dict(message,content=message['content']+
                '\nA small RED cross marks the exact aiming point. Ignore the marker itself; '
                'identify the original material BETWEEN its arms.'))
        else:
            content=[dict(p,image=aim_marker(p['image'])) if p['type']=='image' else dict(p)
                     for p in message['content']]
            if mode in ('crop','crop-feedback-last','target-detail'):
                h,w=current.shape[:2];half=min(h,w)*3//14
                if mode=='target-detail':half=min(h,w)//14
                crop=current[h//2-half:h//2+half,w//2-half:w//2+half]
                crop=np.asarray(Image.fromarray(crop).resize((w,h),Image.Resampling.NEAREST))
                position=(max(i for i,p in enumerate(content) if p['type']=='image')+1
                          if mode=='crop-feedback-last' else len(content)-1)
                content[position:position]=[
                    dict(type='text',text='Magnified center of the CURRENT image, same moment, NOT a new observation. '
                         'This close-up helps identify what the axe is actually aimed at:'),
                    dict(type='image',image=aim_marker(crop))]
            output.append(dict(message,content=content))
    return output
