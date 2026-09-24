import numpy as np
from pipelines.probe import messages_for
from pipelines.probe_vision import aim_marker, aim_messages


def test_marker_preserves_original_and_exact_center():
    a=np.zeros((224,224,3),dtype=np.uint8)
    b=aim_marker(a)
    assert not a.any()
    assert not b[110:115,110:115].any()
    assert b.any()


def test_crop_is_extra_observation_detail_not_extra_inference():
    a=np.zeros((224,224,3),dtype=np.uint8)
    a[100:124,100:124]=77
    m=messages_for(a,a,'system')
    assert aim_messages(m,a,'plain') is m
    modified=aim_messages(m,a,'crop')
    images=[p['image'] for p in modified[1]['content'] if p['type']=='image']
    assert len(images)==3
    assert all(x.shape==(224,224,3) for x in images)
    assert np.all(images[-1][112,112]==77)
    assert len([p for p in m[1]['content'] if p['type']=='image'])==2
    assert modified[1]['content'][-1]['text']=='Choose the next action.'


def test_feedback_can_follow_all_visual_details():
    a=np.zeros((224,224,3),dtype=np.uint8)
    m=messages_for(a,a,'system',last_action='dig',feedback='three failed digs')
    modified=aim_messages(m,a,'crop-feedback-last')[1]['content']
    last_image=max(i for i,p in enumerate(modified) if p['type']=='image')
    feedback=next(i for i,p in enumerate(modified) if 'three failed digs' in p.get('text',''))
    assert feedback>last_image


def test_current_large_keeps_only_current_frame_and_public_feedback():
    previous=np.zeros((224,224,3),dtype=np.uint8)
    current=np.full_like(previous,77)
    m=messages_for(previous,current,'system',last_action='left',feedback='one failed dig')
    out=aim_messages(m,current,'current-large')
    images=[p['image'] for p in out[1]['content'] if p['type']=='image']
    assert len(images)==1 and images[0].shape==(672,672,3)
    assert np.all(images[0]==77)
    assert any('one failed dig' in p.get('text','') for p in out[1]['content'])
    assert not previous.any() and np.all(current==77)
    assert len([p for p in m[1]['content'] if p['type']=='image'])==2


def test_target_detail_does_not_invent_new_pixels_at_aim():
    a=np.full((224,224,3),11,dtype=np.uint8)
    a[96:128,96:128]=77
    out=aim_messages(messages_for(a,a,'system'),a,'target-detail')
    images=[p['image'] for p in out[1]['content'] if p['type']=='image']
    assert len(images)==3 and np.all(images[-1][112,112]==77)
    assert np.all(images[-1][0,0]==77)
