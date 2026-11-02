"""Run after pip install -r requirements-enhancement.txt to install official weights."""
from pathlib import Path
import hashlib
import sys
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.enhancement import WEIGHTS,ROOT

def main():
    target=ROOT/'.models/realesrgan';target.mkdir(parents=True,exist_ok=True)
    for name,digest in WEIGHTS.items():
        path=target/name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==digest:continue
        temporary=path.with_suffix('.download')
        try:
            urllib.request.urlretrieve('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/'+name,temporary)
            if hashlib.sha256(temporary.read_bytes()).hexdigest()!=digest:raise RuntimeError('Model checksum mismatch: '+name)
            temporary.replace(path)
        finally:temporary.unlink(missing_ok=True)
    print('Verified official Real-ESRGAN model weights installed.')

if __name__=='__main__':main()
