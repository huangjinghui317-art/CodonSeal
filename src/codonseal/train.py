"""Rebuild the 192-task, two-token CodonSeal WPN using the pinned Open-R1 splits."""
import argparse
import fcntl
import io
import json
import math
from pathlib import Path
import re
import sqlite3

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np
from tqdm import tqdm

from .backend import MLXBackend
from .util import chunks, digest, dump, encode
from .watermark import Tiny, windows

ROOT = Path(__file__).resolve().parents[2]
EPOCHS, PATIENCE = 10,3


def geometry(backend,path):
    if path.is_file():
        return mx.load(str(path))
    units = []
    for start in tqdm(range(0,backend.size,1024),desc='Build frozen vocabulary geometry'):
        vectors = backend.embedding(mx.arange(start,min(start+1024,backend.size))).astype(mx.float32)
        vectors = vectors/mx.maximum(mx.linalg.norm(vectors,axis=-1,keepdims=True),1e-8)
        units.append(np.array(vectors.astype(mx.float16)))
    unit = np.concatenate(units)
    mean = unit.mean(0,dtype=np.float64).astype(np.float32)
    projection = np.random.default_rng(42).normal(size=(unit.shape[1],64)).astype(np.float32)/8
    bank = np.concatenate([(v.astype(np.float32)@projection).astype(np.float16) for v in units])
    result = dict(units=mx.array(unit),mean=mx.array(mean),bank=mx.array(bank))
    tmp = path.with_name('geometry.partial.safetensors')
    mx.save_safetensors(str(tmp),result)
    tmp.replace(path)
    return result


def corpus(local,projected):
    manifest = json.loads((ROOT/'configs/openr1_assets.json').read_text())
    splits = json.loads((ROOT/'configs/openr1_splits.json').read_text())
    wanted = set(splits['train']+splits['validation'])
    assert len(wanted)==240 and not wanted.intersection(splits['development'])
    found = {}
    if projected:
        # This option accepts a local column projection of the same public shards.
        files = [projected/name for name in sorted(manifest['projected_sha256'])]
        for path in files:
            assert digest(path)==manifest['projected_sha256'][path.name], f'Modified projected shard: {path}'
        batches = (json.loads(line) for path in files
                   for line in path.read_text().split('\n') if line)
    else:
        import pyarrow.parquet as pq
        files = [local/entry['path'] for entry in manifest['assets']]
        for path,entry in zip(files,manifest['assets'],strict=True):
            assert digest(path)==entry['sha256'], f'Modified Open-R1 artifact: {path}'
        batches = (row for path in files for b in pq.ParquetFile(path).iter_batches(
            columns=['problem_id','problem_statement','gold_standard_solution']) for row in b.to_pylist())
    for row in batches:
        i = 'OpenR1/'+str(row['problem_id'])
        if i not in wanted or i in found:
            continue
        text = row['gold_standard_solution']
        code = re.findall(r'```(?:python|py)?[ \t]*\n(.*?)```',text,flags=re.S|re.I)
        found[i] = dict(id=i,prompt=row['problem_statement'].strip()+'\n\nReturn Python code only.',
                        reference=(code[0] if code else text).strip()+'\n')
    if set(found)!=wanted:
        raise ValueError(f'Missing {len(wanted-set(found))} pinned Open-R1 problems.')
    return [dict(found[i],split=split) for split in ('train','validation') for i in splits[split]]


def teacher_examples(backend, geometry, db, rows, progress):
    for row in tqdm(rows, desc='Open-R1 teacher labels'):
        if db.execute('select 1 from labeled where id=?', (row['id'],)).fetchone():
            continue
        ids = backend.tokenizer.encode(row['reference'], add_special_tokens=False)
        # Gold code is teacher-forced under its own external problem.
        prefix = backend.prompt(row['prompt'])
        hidden = backend.model.model(mx.array([prefix + ids[:-1]], dtype=mx.int32))[0, len(prefix)-1:]
        contexts = windows(ids)
        head = backend.embedding.as_linear if backend.model.args.tie_word_embeddings else backend.model.lm_head
        for start in range(0, len(ids), 32):
            h = hidden[start:start+32]
            raw = head(h).astype(mx.float32)
            top = mx.argpartition(-raw, 63, axis=-1)[:, :64]
            v = mx.take_along_axis(raw, top, axis=-1)
            p = mx.softmax(v/.8, axis=-1)
            mean = (geometry['units'][top].astype(mx.float32)*p[:, :, None]).sum(1)
            logp = raw - mx.logsumexp(raw, axis=-1, keepdims=True)
            entropy = -(mx.exp(logp)*logp).sum(-1)
            counts = (v >= v.max(-1, keepdims=True)-2.5).sum(-1)
            arrays = [np.array(x) for x in (mean.astype(mx.float16), entropy, counts, h.astype(mx.float16))]
            for j, (vec, ent, count, state) in enumerate(zip(*arrays)):
                t = start+j
                if t < 2:
                    continue
                ngram = encode(ids[t-2:t+1])
                if db.execute('select 1 from teacher where ngram=?', (ngram,)).fetchone():
                    continue
                packed = io.BytesIO()
                np.savez_compressed(packed, x=contexts[t], count=np.int32(count), mean=vec,
                                    hidden=state, target=np.int32(ids[t]), entropy=np.float32(ent))
                db.execute('insert into teacher(ngram,problem,split,payload) values(?,?,?,?)',
                           (ngram, row['id'], row['split'], packed.getvalue()))
        db.execute('insert into labeled values(?)', (row['id'],))
        db.commit()
        progress()
    mx.clear_cache()


def arrays(db):
    result = {}
    for split,n in (('train',13577),('validation',2983)):
        x = np.empty((n,16),dtype=np.int32)
        count = np.empty(n,dtype=np.int32)
        means = np.empty((n,1536),dtype=np.float16)
        rows = db.execute('select payload from teacher where split=? order by id',(split,))
        seen = 0
        for i,(payload,) in enumerate(rows):
            with np.load(io.BytesIO(payload)) as data:
                x[i],count[i],means[i] = data['x'],data['count'],data['mean']
            seen += 1
        assert seen==n, (split,seen,n)
        result[split] = x,count,means
    return result


def fit(data,geo,directory):
    directory.mkdir(exist_ok=True)
    state_file = directory/'training.json'
    mx.random.seed(42)
    net = Tiny(geo['bank'],2)
    count = data['train'][1]
    freq = np.bincount(np.where(count<=2,0,np.where(count==3,1,2)),minlength=3)
    weights = np.sqrt(freq.sum()/np.maximum(3*freq,1))
    weights = mx.array((weights/weights.mean()).astype(np.float32))

    def objective(model,x,count,expected):
        logits,pred,mean = model(x)
        label = mx.where(count<=2,0,mx.where(count==3,1,2))
        ce = nn.losses.cross_entropy(logits,label,label_smoothing=.05,reduction='none')
        return (ce*weights[label]).mean()+.25*mx.mean((pred-mx.log1p(count))**2)+.5*mx.mean(
            mx.sum((mean-expected.astype(mx.float32))**2,axis=-1))

    opt = optim.AdamW(.0003,weight_decay=.01)
    state = dict(next_epoch=0,step=0,best=1e30,stale=0,history=[])
    if state_file.exists():
        state = json.loads(state_file.read_text())
        previous = directory/f"epoch-{state['next_epoch']}"
        net.load_weights(str(previous/'weights.safetensors'))
        saved = mx.load(str(previous/'optimizer.safetensors'))
        rng = saved.pop('__rng__')
        opt.state = tree_unflatten(list(saved.items()))
        mx.random.state = [rng]

    batch_size = 128
    gradient = nn.value_and_grad(net,objective)
    steps = EPOCHS*math.ceil(len(count)/batch_size)
    for epoch in range(state['next_epoch'],EPOCHS):
        if state['stale']>=PATIENCE:
            break
        net.train()
        order = np.random.default_rng(42+epoch).permutation(len(count))
        losses = []
        for ix in tqdm(list(chunks(order,batch_size)),desc=f'WPN epoch {epoch+1}/{EPOCHS}'):
            opt.learning_rate = .0003*min(1.,(state['step']+1)/50)*.5*(
                1+math.cos(math.pi*state['step']/max(1,steps)))
            value,grads = gradient(net,*[mx.array(a[ix]) for a in data['train']])
            grads,_ = optim.clip_grad_norm(grads,1.)
            opt.update(net,grads)
            mx.eval(net.parameters(),opt.state,value)
            losses.append(float(value))
            state['step'] += 1
        net.eval()
        values = []
        for ix in chunks(np.arange(len(data['validation'][0])),batch_size):
            value = objective(net,*[mx.array(a[ix]) for a in data['validation']])
            mx.eval(value)
            values.append(float(value)*len(ix))
        val = sum(values)/len(data['validation'][0])
        state['history'].append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),validation_loss=val))
        if val<state['best']:
            state.update(best=val,stale=0,best_epoch=epoch+1)
        else:
            state['stale'] += 1
        state['next_epoch'] = epoch+1
        dest = directory/f'epoch-{epoch+1}'
        dest.mkdir(exist_ok=True)
        net.save_weights(str(dest/'weights.safetensors'))
        saved = dict(tree_flatten(opt.state))
        saved['__rng__'] = mx.random.state[0]
        mx.save_safetensors(str(dest/'optimizer.safetensors'),saved)
        dump(state_file,state)
    chosen = directory/f"epoch-{state['best_epoch']}"/'weights.safetensors'
    net.load_weights(str(chosen))
    net.save_weights(str(directory/'best.safetensors'))
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local',type=Path,default=Path('local'))
    parser.add_argument('--projected-openr1',type=Path,help='Existing JSONL projections of the pinned training shards.')
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    out = args.out or args.local/'wpn'
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rebuild(args,out)


def rebuild(args,out):
    signature = {str(p.relative_to(ROOT)):digest(p) for p in (
        ROOT/'configs/openr1_assets.json',ROOT/'configs/openr1_splits.json',
        ROOT/'configs/assets.json',Path(__file__),Path(__file__).with_name('watermark.py'))}
    state = out/'run.json'
    if state.exists() and json.loads(state.read_text())!=signature:
        raise ValueError('Training configuration changed. Resume with the original sources or choose a fresh output directory.')
    dump(state,signature)
    for entry in json.loads((ROOT/'configs/assets.json').read_text())['assets']:
        if entry['kind']=='model':
            assert digest(args.local/entry['path'])==entry['sha256']
    rows = corpus(args.local,args.projected_openr1)
    mx.set_cache_limit(128*1024**2)
    backend = MLXBackend(args.local/'model',b'teacher only')
    geo = geometry(backend,out/'geometry.safetensors')
    with sqlite3.connect(out/'teacher.sqlite') as db:
        db.executescript("""
            create table if not exists teacher(id integer primary key,ngram text unique,problem text,split text,payload blob);
            create table if not exists labeled(id text primary key);
        """)
        teacher_examples(backend,geo,db,rows,lambda:None)
        assert set(i for i, in db.execute('select id from labeled'))=={row['id'] for row in rows}
        data = arrays(db)
    state = fit(data,geo,out/'tiny2')
    actual = dict(checkpoint_sha256=digest(out/'tiny2/best.safetensors'),
                  geometry_sha256=digest(out/'geometry.safetensors'),best_epoch=state['best_epoch'],
                  training_positions=len(data['train'][0]),validation_positions=len(data['validation'][0]))
    dump(out/'training_result.json',actual)
    print(json.dumps(actual,indent=2))


if __name__=='__main__':
    main()
