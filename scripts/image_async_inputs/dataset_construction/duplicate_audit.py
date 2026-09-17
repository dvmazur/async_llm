"""Offline cross-source duplicate candidates; never silently exclude records."""
import argparse
from difflib import SequenceMatcher
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageChops, ImageStat
from pipeline import read, records, save


def audit(builds, prior, output):
    output.mkdir(parents=True, exist_ok=True)
    items=[]
    for root in [prior, *sorted(p for p in builds.iterdir() if p.is_dir())]:
        for r in records(root):
            if r['status'] != 'accepted':
                continue
            path=root/r['normalization']['images']['after']['path']
            with Image.open(path) as im:
                rgb=im.convert('RGB').resize((128,128))
                gray=rgb.convert('L').resize((17,16))
                pixels=list(gray.getdata())
                bits=sum((pixels[y*17+x]>pixels[y*17+x+1]) << (y*16+x)
                         for y in range(16) for x in range(16))
            items.append((root,r,path,rgb,bits))
    pairs=[]
    for i,a in enumerate(items):
        for b in items[i+1:]:
            if a[1]['source']['dataset']==b[1]['source']['dataset']:
                continue
            distance=(a[4]^b[4]).bit_count()
            qa=a[1]['source']['question'].lower(); qb=b[1]['source']['question'].lower()
            similarity=SequenceMatcher(None,qa,qb).ratio()
            if distance>22 and similarity<0.94:
                continue
            error=sum(ImageStat.Stat(ImageChops.difference(a[3],b[3])).mean)/3
            if error>20 and similarity<0.94:
                continue
            ids=[a[1]['id'],b[1]['id']]
            canvas=Image.new('RGB',(1600,650),'#ddd'); draw=ImageDraw.Draw(canvas)
            for side,item in enumerate((a,b)):
                with Image.open(item[2]) as im:
                    thumb=ImageOps.contain(im.convert('RGB'),(790,610))
                    canvas.paste(thumb,(side*800+(800-thumb.width)//2,35+(610-thumb.height)//2))
                draw.text((side*800+5,5),item[1]['source']['dataset']+' '+item[1]['id'],fill='black')
            preview=output/('_'.join(ids)+'.png');canvas.save(preview)
            pairs.append({'ids':ids,'datasets':[a[1]['source']['dataset'],b[1]['source']['dataset']],
                          'questions':[qa,qb],'dhash_distance':distance,'pixel_mae':error,
                          'question_similarity':similarity,'preview':str(preview.resolve())})
    save(output/'candidates.json',pairs)
    print({'accepted':len(items),'cross_source_candidates':len(pairs)})
    for p in pairs: print({k:p[k] for k in ('ids','dhash_distance','pixel_mae','question_similarity')})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--builds',type=Path,default=Path('work/v3_builds'))
    p.add_argument('--prior',type=Path,default=Path('work/v2_assembled'))
    p.add_argument('--output',type=Path,default=Path('work/v3_duplicate_audit'))
    a=p.parse_args();audit(a.builds,a.prior,a.output)
