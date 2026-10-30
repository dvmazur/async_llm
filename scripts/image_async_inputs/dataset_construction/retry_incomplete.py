"""Explicit ONE retry of a truncated text response; preserve the original attempt."""
import argparse
from pathlib import Path
import gateway
from image_edit_test import data_url
from pipeline import read,save,require,digest

def retry(path):
    old=read(path)
    require(old['choices'][0]['finish_reason']=='length','Only truncated responses may be retried')
    out=path.with_suffix('.retry.json')
    require(not out.exists(),'One retry only; saved retry exists')
    request=read(path.with_suffix('.request.json'))
    require(request['model']=='google/gemini-3.8-flash' and request['max_tokens']==4096,'Unexpected request')
    for im in request['images']:
        require(digest(Path(im['path']))==im['sha256'],'Image changed')
    settings={k:v for k,v in request.items() if k not in ('prompt','images')}
    settings['max_tokens']=8192
    save(out.with_suffix('.request.json'),{**request,'max_tokens':8192})
    response=gateway.request({**settings,'messages':[{'role':'user','content':[
        {'type':'text','text':request['prompt']},*[{'type':'image_url','image_url':{'url':data_url(Path(im['path']))}} for im in request['images']]]}]})
    save(out,response)
    print(path, response['choices'][0]['finish_reason'], response.get('usage',{}).get('cost'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('paths',type=Path,nargs='+')
    args=p.parse_args()
    for path in args.paths: retry(path)
