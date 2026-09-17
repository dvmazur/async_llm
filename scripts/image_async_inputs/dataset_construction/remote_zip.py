"""Read public ZIP members with HTTP ranges; verify range and archive identity."""
import io
import re
import urllib.request
import urllib.error
import time
import zipfile


class RemoteZipFile(io.RawIOBase):
    def __init__(self,url):
        self.url=url
        self.position=0
        self.buffer_start=0
        self.buffer=b''
        req=urllib.request.Request(url,headers={'Range':'bytes=-1','Accept-Encoding':'identity'})
        with urllib.request.urlopen(req,timeout=90) as response:
            if response.status!=206:
                raise ValueError('Server does not support byte ranges')
            self.length=int(response.headers['Content-Range'].split('/')[-1])
            self.etag=response.headers.get('ETag')
            response.read()

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        position=offset if whence==0 else self.position+offset if whence==1 else self.length+offset
        if position<0:raise ValueError('Negative seek')
        self.position=position
        return position

    def read(self,size=-1):
        size=self.length-self.position if size<0 else min(size,self.length-self.position)
        if size<=0:return b''
        if self.buffer_start <= self.position and self.position+size <= self.buffer_start+len(self.buffer):
            offset=self.position-self.buffer_start
            self.position+=size
            return self.buffer[offset:offset+size]
        start=self.position;end=start+size-1
        fetch_size=min(max(size,256*1024),self.length-start)
        end=start+fetch_size-1
        headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'}
        if self.etag and not self.etag.startswith('W/'):
            headers['If-Match']=self.etag
        req=urllib.request.Request(self.url,headers=headers)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req,timeout=120) as response:
                    if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.length}':
                        raise ValueError('Range or archive identity mismatch')
                    if self.etag and response.headers.get('ETag')!=self.etag:
                        raise ValueError('Archive ETag changed')
                    data=response.read()
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in (500,502,503,504) or attempt==2:raise
                time.sleep(1)  # Only idempotent public file reads, never paid model calls.
        if len(data)!=fetch_size:raise ValueError('Truncated range response')
        self.buffer_start=start;self.buffer=data
        self.position+=size
        return data[:size]


URLS={
 'mapqa':'https://drive.usercontent.google.com/download?id=1ul_FcgoHbcm5txmYEr9IQtkN6e8sRZJy&export=download&confirm=t',
 'clevr':'https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip'}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset',choices=URLS)
    a=p.parse_args()
    remote=RemoteZipFile(URLS[a.dataset])
    with zipfile.ZipFile(remote) as archive:
        print('archive',remote.length,remote.etag,'entries',len(archive.infolist()))
        print([(x.filename,x.file_size,x.compress_size) for x in archive.infolist()
               if x.filename.endswith('.json')][:20])
