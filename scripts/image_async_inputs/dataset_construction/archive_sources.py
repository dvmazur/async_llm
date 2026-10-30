"""Original MapQA/CLEVR image acquisition from official archives via byte ranges."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from pipeline import digest, import_sources, read, require, save
from remote_zip import RemoteZipFile, URLS

META_HASHES={
 'mapqa':{'test-QA.json':'ce3db4b214f810108f13a81faa195a7beadca0602f3d73bf303176d43da6f9c9'},
 'clevr':{'CLEVR_val_questions.json':'fd9ff46c9dc842f62322158695d07d33e2c61034ffed51fc4bcb214e0adb5f51',
          'CLEVR_val_scenes.json':'f53e27a9463d00534a857aac37601b7166749f811c566ce0dca8d32bf77e0585'}}


def prepare(name,cache,root,limit,offset):
    require(not root.exists(),'Use a fresh workspace')
    lock=read(cache/'archive_lock.json')
    for filename,checksum in META_HASHES[name].items():
        require(digest(cache/filename)==checksum,'Metadata checksum mismatch')
    remote=RemoteZipFile(URLS[name])
    require(remote.length==lock['bytes'] and remote.etag==lock['etag'],'Archive identity changed')
    if name=='mapqa':
        rows=read(cache/'test-QA.json')
        rows=[r for r in rows if r['question_type']!='retrieval']
        image_key='map_id'; id_key='question_id'
        scenes={}
    else:
        rows=read(cache/'CLEVR_val_questions.json')['questions']
        rows=[r for r in rows if len(r['program'])>=10 and str(r['answer']) not in ('yes','no')]
        image_key='image_filename'; id_key='question_index'
        scenes={r['image_index']:r for r in read(cache/'CLEVR_val_scenes.json')['scenes']}
    rows.sort(key=lambda r:hashlib.sha256(f"42:{r[id_key]}".encode()).hexdigest())
    unique=[];seen=set()
    for row in rows:
        if row[image_key] not in seen:
            unique.append(row);seen.add(row[image_key])
    (root/'records').mkdir(parents=True)
    staging=root/'sources';staging.mkdir()
    sources=[]; members={}
    with zipfile.ZipFile(remote) as archive:
        names=set(archive.namelist())
        def member(filename):
            require(filename in names,'Missing archive member: '+filename)
            info=archive.getinfo(filename)
            target=cache/'members'/filename
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(archive.read(filename))  # ZipFile checks decompression CRC.
            checksum=digest(target)
            members[filename]={'sha256':checksum,'crc32':info.CRC,'bytes':info.file_size}
            require(target.stat().st_size==info.file_size,'Cached member size mismatch')
            import zlib
            require(zlib.crc32(target.read_bytes())==info.CRC,'Cached member CRC mismatch')
            return target
        for row in unique[offset:offset+limit]:
            sid=str(row[id_key])
            if name=='mapqa':
                img=f"MapQA_U/images/{row[image_key]}"
                original=dict(row)
                meta=f"MapQA_U/metadata/{Path(row[image_key]).stem}.json"
                if meta in names: original['map_metadata']=read(member(meta))
                dataset='OSU-slatelab/MapQA-U';split='test'
                category='map_'+row['question_type']
                url='https://github.com/OSU-slatelab/MapQA'
                license='MapQA CC-BY-SA-4.0; retain original KFF content attribution; local evaluation only'
            else:
                img=f"CLEVR_v1.0/images/val/{row[image_key]}"
                original={**row,'scene':scenes[row['image_index']]}
                dataset='CLEVR/v1.0';split='val'
                category='synthetic_compositional_'+row['program'][-1]['function']
                url='https://cs.stanford.edu/people/jcjohns/clevr/'
                license='CLEVR v1.0 CC-BY-4.0; synthetic source scenes; retain attribution'
            image=member(img)
            answer=row['answer']
            if isinstance(answer,list):answer='; '.join(str(x) for x in answer)
            sources.append({'dataset':dataset,'source_id':sid,'source_split':split,
                'category':category,'question':row['question'],'answer':str(answer),
                'image':str(image.resolve()),'source_url':url,'license':license,
                'source_revision':f"archive-bytes-{remote.length}-questions-sha256-{next(iter(META_HASHES[name].values()))}",
                'original':original,'selection_seed':42,'selection_offset':offset})
            print(f'Acquired {name} {sid}',flush=True)
    manifest=staging/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in sources))
    import_sources(root,manifest)
    save(root/'selection.json',{'archive':lock,'metadata_sha256':META_HASHES[name],
                              'members':members,'offset':offset,'count':len(sources)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',choices=URLS)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--limit',type=int,default=30)
    p.add_argument('--offset',type=int,default=0)
    a=p.parse_args()
    prepare(a.dataset,Path('work/'+a.dataset+'_cache'),a.workspace.resolve(),a.limit,a.offset)
