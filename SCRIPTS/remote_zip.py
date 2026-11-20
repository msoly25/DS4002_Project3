"""Bounded HTTP-range reader for the authors' split ZIP files (no full archives)."""
import html
import re
import struct
import threading
import time
import zlib
from pathlib import Path
import requests

class DriveRanges:
    def __init__(self):
        self.urls = {}
        self.lock = threading.Lock()
        self.bytes_received = 0

    def url(self, file_id):
        with self.lock:
            if file_id not in self.urls:
                with requests.get('https://drive.google.com/uc', params={'export':'download','id':file_id}, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    if 'text/html' in r.headers.get('Content-Type',''):
                        page = r.text
                        match = re.search(r'<form[^>]+action="([^"]+)', page)
                        if not match:
                            raise RuntimeError('Drive download unavailable: '+file_id)
                        action = html.unescape(match.group(1))
                        params = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"',page))
                        self.urls[file_id] = requests.Request('GET',action,params=params).prepare().url
                    else:
                        self.urls[file_id] = r.url
            return self.urls[file_id]

    def get(self, file_id, start, end):
        expected = end-start+1
        for attempt in range(5):
            try:
                with requests.get(self.url(file_id), headers={'Range':f'bytes={start}-{end}'}, stream=True, timeout=90) as r:
                    if r.status_code != 206:
                        raise RuntimeError(f'Expected partial response, got {r.status_code}; full download refused')
                    cr = r.headers.get('Content-Range','')
                    if not cr.startswith(f'bytes {start}-{end}/'):
                        raise RuntimeError('Incorrect Content-Range: '+cr)
                    data = b''.join(r.iter_content(1024*1024))
                    if len(data) != expected:
                        raise RuntimeError('Incomplete range')
                    with self.lock:
                        self.bytes_received += len(data)
                    return data
            except (requests.RequestException, RuntimeError):
                if attempt == 4: raise
                time.sleep(2**attempt)

class SplitZip:
    def __init__(self, name, files, client, cache_dir=None):
        self.name, self.client = name, client
        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path(__file__).resolve().parents[1] / "DATA" / "original" / "cache"
        # Duplicate filenames exist in SD1.5; choose a stable file ID.
        unique = {}
        for f in sorted(files,key=lambda f:f['id']): unique.setdefault(f['title'],f)
        self.parts = sorted([f for f in unique.values() if re.search(r'\.z\d+$', f['title'])],key=lambda f:int(f['title'].rsplit('.z',1)[1]))
        self.parts += [f for f in unique.values() if f['title'].endswith('.zip')]
        self.entries = []

    def read(self,disk,offset,size):
        chunks=[]
        while size:
            f=self.parts[disk]; take=min(size,int(f['size'])-offset)
            if take <= 0: raise ValueError('Invalid split offset')
            chunks.append(self.client.get(f['id'],offset,offset+take-1))
            size-=take; disk+=1; offset=0
        return b''.join(chunks)

    def index(self):
        cache=self.cache_dir/f'{self.name}_central.bin'; cache.parent.mkdir(parents=True,exist_ok=True)
        last=self.parts[-1]; size=int(last['size'])
        if cache.exists():
            central=cache.read_bytes()
        else:
            tail=self.client.get(last['id'],size-65536,size-1)
            pos=tail.rfind(b'PK\x06\x06')
            if pos>=0:
                _,_,_,_,disk,cd_disk,n_disk,n_total,cd_size,cd_offset=struct.unpack_from('<4sQHHIIQQQQ',tail,pos)
            else:
                pos=tail.rfind(b'PK\x05\x06')
                _,disk,cd_disk,n_disk,n_total,cd_size,cd_offset,_=struct.unpack_from('<4s4H2IH',tail,pos)
            if disk!=len(self.parts)-1: raise ValueError('Incomplete multipart inventory')
            central=self.read(cd_disk,cd_offset,cd_size); cache.write_bytes(central)
        pos=0
        while pos<len(central):
            vals=struct.unpack_from('<4s6H3I5H2I',central,pos)
            if vals[0]!=b'PK\x01\x02': raise ValueError('Invalid central directory')
            _,made,needed,flags,method,tm,dt,crc,csize,usize,nlen,xlen,clen,disk,ia,ea,offset=vals
            name=central[pos+46:pos+46+nlen].decode('utf-8' if flags&2048 else 'cp437')
            extra=central[pos+46+nlen:pos+46+nlen+xlen]; ep=0
            while ep+4<=len(extra):
                tag,length=struct.unpack_from('<HH',extra,ep); payload=extra[ep+4:ep+4+length]; ep+=4+length
                if tag==1:
                    q=0
                    if usize==0xffffffff: usize=struct.unpack_from('<Q',payload,q)[0];q+=8
                    if csize==0xffffffff: csize=struct.unpack_from('<Q',payload,q)[0];q+=8
                    if offset==0xffffffff: offset=struct.unpack_from('<Q',payload,q)[0];q+=8
                    if disk==0xffff: disk=struct.unpack_from('<I',payload,q)[0]
            self.entries.append(dict(source_path=name,disk=disk,offset=offset,compressed_bytes=csize,uncompressed_bytes=usize,crc32=crc,method=method))
            pos+=46+nlen+xlen+clen
        return self.entries

    def extract(self,e):
        # Local filename and extra fields normally fit in this allowance; expand if needed.
        raw=self.read(e['disk'],e['offset'],30+len(e['source_path'].encode('utf-8'))+256+e['compressed_bytes'])
        if raw[:4]!=b'PK\x03\x04': raise ValueError('Bad local header')
        n,x=struct.unpack_from('<HH',raw,26); start=30+n+x
        if len(raw)<start+e['compressed_bytes']:
            raw=self.read(e['disk'],e['offset'],start+e['compressed_bytes'])
        compressed=raw[start:start+e['compressed_bytes']]
        data=compressed if e['method']==0 else zlib.decompress(compressed,-15) if e['method']==8 else None
        if data is None: raise ValueError('Unsupported compression')
        if len(data)!=e['uncompressed_bytes'] or zlib.crc32(data)!=e['crc32']: raise ValueError('ZIP integrity check failed')
        return data
