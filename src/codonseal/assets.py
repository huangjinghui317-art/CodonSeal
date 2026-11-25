"""Fetch the pinned public model and datasets without storing them in Git."""
import argparse
import json
from pathlib import Path
import subprocess

from .util import digest

ROOT = Path(__file__).resolve().parents[2]


def fetch(entry,root):
    target = root/entry['path']
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and digest(target)==entry['sha256']:
        print(f'Verified {target}')
        return
    tmp = target.with_suffix(target.suffix+'.part')
    if 'contents' in entry:
        tmp.write_text(entry['contents'],encoding='utf-8')
    else:
        for url in entry['urls']:
            size = tmp.stat().st_size if tmp.exists() else 0
            if size and digest(tmp)==entry['sha256']:
                break
            args = ['curl','--fail','--location','--show-error','--progress-bar','--connect-timeout','15',
                    '--max-time','900','--retry','1','--continue-at','-','-o',str(tmp),url]
            done = subprocess.run(args,check=False).returncode==0
            if tmp.exists() and digest(tmp)==entry['sha256']:
                break
            if tmp.exists() and (done or (entry.get('bytes') and tmp.stat().st_size>=entry['bytes'])):
                tmp.unlink()
    if not tmp.exists() or digest(tmp)!=entry['sha256']:
        raise RuntimeError(f'Download failed or did not match SHA256 for {target}. Check the connection or place a verified file there.')
    tmp.replace(target)
    print(f'Verified {target}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('local'))
    parser.add_argument('--with-training',action='store_true',help='Also fetch the four Open-R1 training shards.')
    args = parser.parse_args()
    entries = json.loads((ROOT/'configs/assets.json').read_text())['assets']
    if args.with_training:
        entries += json.loads((ROOT/'configs/openr1_assets.json').read_text())['assets']
    for entry in entries:
        fetch(entry,args.directory)


if __name__=='__main__':
    main()
