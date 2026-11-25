"""Reproducible, resume-safe HumanEval evaluation of the three release methods."""
import argparse
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from tqdm import tqdm

from .grading import grade
from .util import chunks, digest, dump, encode

ROOT = Path(__file__).resolve().parents[2]
METHODS = ('unmarked','CodonSeal-oracle','CodonSeal')
SCORED = METHODS[1:]
PREFIX = 'Complete the following Python task. Return the full function definition and any required imports. Output Python code only.\n\n'


def load_questions(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows)==164 and len({r['task_id'] for r in rows})==164
    return [dict(id=r['task_id'],prompt=PREFIX+r['prompt'],test=r['test'],entry_point=r['entry_point']) for r in rows]


def run(model,humaneval,geometry,checkpoint,output,only_seeds=None,key_file=None):
    import mlx.core as mx
    from .watermark import BodyDetector, ReleaseBackend, Tiny

    protocol = json.loads((ROOT/'configs/protocol.json').read_text())
    seeds = only_seeds if only_seeds is not None else protocol['seeds']
    if not seeds or len(seeds)!=len(set(seeds)):
        raise ValueError('At least one distinct seed is required.')
    questions = load_questions(humaneval)
    spec = json.loads((ROOT/'configs/assets.json').read_text())['assets']
    hashes = {row['path']:row['sha256'] for row in spec}
    assert digest(humaneval)==hashes['data/humaneval/HumanEval.jsonl.gz']
    model_files = {p.removeprefix('model/'):h for p,h in hashes.items() if p.startswith('model/')}
    for file,sha in model_files.items():
        assert digest(model/file)==sha,file
    key = key_file.read_bytes() if key_file else hashlib.sha256(protocol['public_key_label'].encode()).digest()
    geometry_sha = digest(geometry)
    checkpoint_sha = digest(checkpoint)
    if (geometry_sha!=protocol['expected_geometry_sha256'] or checkpoint_sha!=protocol['expected_checkpoint_sha256']):
        print('Using locally rebuilt assets (checksum differs from archived evaluation artifacts).',flush=True)
    signature = dict(methods=list(METHODS),seeds=seeds,protocol=protocol,
         software={p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*')) if p.suffix in ('.py','.sb') and p.name not in {'train.py','export.py','assets.py'}},
         questions=[r['id'] for r in questions],assets=dict(model=model_files,geometry=geometry_sha,checkpoint=checkpoint_sha,
             humaneval=hashes['data/humaneval/HumanEval.jsonl.gz']),
         private_key_sha256=digest(key_file) if key_file else None)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest = output/'run.json'
        if manifest.exists() and json.loads(manifest.read_text())!=signature:
            raise ValueError('Resume configuration does not match run.json. Use a fresh output directory.')
        dump(manifest,signature)
        with sqlite3.connect(output/'run.sqlite') as db:
            db.executescript('''
            create table if not exists generations(seed integer, method text, id text, payload text, primary key(seed,method,id));
            create table if not exists scores(seed integer, detector text, label text, id text, score real, primary key(seed,detector,label,id));
            create table if not exists grades(seed integer, method text, id text, grade text, primary key(seed,method,id));
            ''')
            started = time.time()
            target = {'generations':len(seeds)*3*164,'scores':len(seeds)*4*164,'grades':len(seeds)*3*164}
            phase = 'initialization'

            def update(s=None,status='running',error=None):
                nonlocal phase
                if s:
                    phase = s
                current = {name:db.execute(f'select count(*) from {name}').fetchone()[0] for name in target}
                report = dict(status=status,phase=phase,current=current,target=target,error=error,elapsed_seconds=time.time()-started)
                dump(output/'progress.json',report)
                (output/'progress.html').write_text('<meta charset="utf-8"><meta http-equiv="refresh" content="5">'
                    '<title>CodonSeal HumanEval benchmark</title><h1>CodonSeal HumanEval benchmark</h1>'
                    f'<pre>{json.dumps(report,indent=2)}</pre>',encoding='utf-8')

            try:
                update()
                mx.set_cache_limit(128*1024**2)
                backend = ReleaseBackend(model,key)
                geo = mx.load(str(geometry))
                net = Tiny(geo['bank'],2)
                net.load_weights(str(checkpoint))
                net.eval()
                detector = BodyDetector(net,backend.tokenizer,geo,key)
                for seed in seeds:
                    for method in METHODS:
                        update(f'{seed}: generate {method}')
                        for batch in tqdm(list(chunks(questions,protocol['batch_size'])),desc=phase):
                            todo = [{'id':r['id'],'prompt':r['prompt']} for r in batch if not db.execute(
                                'select 1 from generations where seed=? and method=? and id=?',(seed,method,r['id'])).fetchone()]
                            if not todo:
                                continue
                            if method=='CodonSeal':
                                outputs = backend.sync_generate(todo,detector,False,seed)
                            else:
                                variant = 'baseline' if method=='unmarked' else 'full'
                                outputs = backend.generate([(r,variant) for r in todo],protocol['max_answer_tokens'],seed)
                            db.executemany('insert into generations values(?,?,?,?)',
                                           [(seed,method,r['id'],encode(g)) for r,g in zip(todo,outputs,strict=True)])
                            db.commit()
                            update()
                    generations = {(m,i):json.loads(g) for m,i,g in db.execute(
                        'select method,id,payload from generations where seed=?',(seed,))}
                    for m in SCORED:
                        update(f'{seed}: detect {m}')
                        for r in tqdm(questions,desc=phase):
                            i = r['id']
                            for label,source in (('negative','unmarked'),('positive',m)):
                                if db.execute('select 1 from scores where seed=? and detector=? and label=? and id=?',(seed,m,label,i)).fetchone():
                                    continue
                                g = generations[source,i]
                                if m=='CodonSeal-oracle':
                                    value = backend.replay_qa(dict(g,prompt=r['prompt']),False)['score']
                                else:
                                    value = detector.detect_text(g['text'])['scores']['sqrt_count']
                                db.execute('insert into scores values(?,?,?,?,?)',(seed,m,label,i,value))
                            db.commit()
                            update()
                    for method in METHODS:
                        update(f'{seed}: grade {method}')
                        for row in tqdm(questions,desc=phase):
                            i = row['id']
                            if not db.execute('select 1 from grades where seed=? and method=? and id=?',(seed,method,i)).fetchone():
                                g = generations[method,i]
                                db.execute('insert into grades values(?,?,?,?)',(seed,method,i,grade(row,g['text'])))
                                db.commit()
                                update()
                    print(f'Complete seed {seed}',flush=True)
                for table,total in target.items():
                    assert db.execute(f'select count(*) from {table}').fetchone()[0]==total
                update('complete','complete')
            except BaseException as exc:
                update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'error',error=repr(exc))
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=Path('local/model'))
    parser.add_argument('--humaneval',type=Path,default=Path('local/data/humaneval/HumanEval.jsonl.gz'))
    parser.add_argument('--geometry',type=Path,default=Path('local/wpn/geometry.safetensors'))
    parser.add_argument('--checkpoint',type=Path,default=Path('local/wpn/tiny2/best.safetensors'))
    parser.add_argument('--out',type=Path,default=Path('local/run'))
    parser.add_argument('--seeds',nargs='+',type=int,help='Override the five-seed protocol for local experiments.')
    parser.add_argument('--key-file',type=Path,help='External binary key file for private experiments.')
    args = parser.parse_args()
    run(args.model,args.humaneval,args.geometry,args.checkpoint,args.out,args.seeds,args.key_file)


if __name__=='__main__':
    main()
