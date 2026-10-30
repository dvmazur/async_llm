"""Offline diagnostic thumbnails, never used as dataset images or model inputs."""
import argparse
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
from pipeline import read, records, save


def queue(builds, output, limit):
    output.mkdir(parents=True,exist_ok=True)
    items=[]
    for root in sorted(p for p in builds.iterdir() if p.is_dir()):
        for r in records(root):
            check=root/'evidence'/r['id']/'checks.json'
            if r['status']!='normalized' or not check.exists() or not read(check)['passes_model_checks']:
                continue
            sid=r['id']
            images=r['normalization']['images']
            canvas=Image.new('RGB',(1600,650),'#dddddd')
            draw=ImageDraw.Draw(canvas)
            for i,state in enumerate(('before','after')):
                with Image.open(root/images[state]['path']) as im:
                    thumb=ImageOps.contain(im,(790,615))
                    canvas.paste(thumb,(i*800+(800-thumb.width)//2,30+(615-thumb.height)//2))
                draw.text((i*800+10,10),state+' '+sid,fill='black')
            path=output/(sid+'.png');canvas.save(path)
            items.append({'id':sid,'workspace':str(root.resolve()),'preview':str(path.resolve()),
                'dataset':r['source']['dataset'],'source_id':r['source']['source_id'],
                'question':r['source']['question'],'answer_before':r['proposal']['expected_answer_before'],
                'answer_after':r['source']['answer'],'proposal':r['proposal'],
                'checks':read(check)})
            if len(items)>=limit:break
        if len(items)>=limit:break
    save(output/'queue.json',items)
    for item in items:
        print({k:item[k] for k in ('id','dataset','source_id','question','answer_before','answer_after')})
    print(f'{len(items)} pending visual review; no acceptance decisions made.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--builds',type=Path,default=Path('work/v3_builds'))
    p.add_argument('--output',type=Path,default=Path('work/v3_review_queue'))
    p.add_argument('--limit',type=int,default=12)
    a=p.parse_args();queue(a.builds,a.output,a.limit)
