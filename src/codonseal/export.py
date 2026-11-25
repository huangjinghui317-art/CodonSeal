"""Export aggregate metrics and per-task results for every benchmark seed."""
import argparse
import csv
import json
import math
from pathlib import Path
import sqlite3
import statistics

import numpy as np

from .evaluate import METHODS, SCORED
from .metrics import compare
from .util import digest, dump

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = ['seed','method','n','auroc','ap','tpr_5','fpr_5','tp','fp','passed','pass_at_1','self_ppl']
DISPLAY = {'unmarked':'Unmarked','CodonSeal-oracle':'CodonSeal-oracle','CodonSeal':'CodonSeal'}


def write_csv(path,rows,columns):
    with path.open('w',encoding='utf-8',newline='') as stream:
        out = csv.DictWriter(stream,columns,lineterminator='\n')
        out.writeheader()
        out.writerows(rows)


def summarize(run,output,readme):
    manifest = json.loads((run/'run.json').read_text())
    protocol = json.loads((ROOT/'configs/protocol.json').read_text())
    ids,seeds = manifest['questions'],manifest['seeds']
    assert seeds==protocol['seeds'] and len(ids)==len(set(ids))==164
    # Compare the current protocol fields so archived runs with extra metadata remain exportable.
    assert {key:manifest['protocol'].get(key) for key in protocol}==protocol
    assert manifest['private_key_sha256'] is None, 'The public exporter requires the public experiment key.'
    assert all(digest(Path(__file__).parent/n)==sha for n,sha in manifest['software'].items())
    with sqlite3.connect(f'{(run/"run.sqlite").resolve().as_uri()}?mode=ro',uri=True) as db:
        assert db.execute('pragma quick_check').fetchone()[0]=='ok'
        generations = {(s,m,i):json.loads(p) for s,m,i,p in db.execute('select * from generations')}
        grades = {(s,m,i):g for s,m,i,g in db.execute('select * from grades')}
        scores = {(s,m,l,i):v for s,m,l,i,v in db.execute('select * from scores')}
    expected = {(s,m,i) for s in seeds for m in METHODS for i in ids}
    assert set(generations)==set(grades)==expected
    assert set(scores)=={(s,m,l,i) for s in seeds for m in SCORED for l in ('negative','positive') for i in ids}
    assert all(math.isfinite(x) for x in scores.values())
    n = len(ids)
    metrics = []
    for seed in seeds:
        for method in METHODS:
            items = [generations[seed,method,i] for i in ids]
            assert all(len(g['ids'])==len(g['nll']) and np.isfinite(g['nll']).all() for g in items)
            tokens = sum(len(g['ids']) for g in items)
            passed = sum(grades[seed,method,i]=='pass' for i in ids)
            row = dict(seed=seed,method=method,n=n,passed=passed,pass_at_1=passed/n,
                       self_ppl=math.exp(sum(sum(g['nll']) for g in items)/tokens))
            if method in SCORED:
                d = compare(*[[scores[seed,method,l,i] for i in ids] for l in ('negative','positive')],[])
                row.update(auroc=d['auroc'],ap=d['average_precision'],tpr_5=d['tpr'],fpr_5=d['fpr'],tp=d['tp'],fp=d['fp'])
            metrics.append(row)
    output.mkdir(parents=True,exist_ok=True)
    seed_dir = output/'merix'
    seed_dir.mkdir(exist_ok=True)
    write_csv(seed_dir/'summary.csv',metrics,COLUMNS)
    for seed in seeds:
        rows = []
        for i in ids:
            record = {'task_id':i}
            for m in METHODS:
                g = generations[seed,m,i]
                record[m+'_passed'] = int(grades[seed,m,i]=='pass')
                record[m+'_answer_tokens'] = len(g['ids'])
                record[m+'_answer_nll'] = sum(g['nll'])
                if m in SCORED:
                    for l in ('negative','positive'):
                        record[m+'_'+l+'_score'] = scores[seed,m,l,i]
            rows.append(record)
        write_csv(seed_dir/f'seed_{seed}.csv',rows,list(rows[0]))

    aggregate = {}
    for m in METHODS:
        rows = [r for r in metrics if r['method']==m]
        aggregate[m] = {}
        for k in ('auroc','ap','tpr_5','fpr_5','pass_at_1','self_ppl'):
            values = [r[k] for r in rows if k in r]
            if values:
                aggregate[m][k] = dict(mean=statistics.mean(values),sd=statistics.stdev(values))
    dump(output/'summary.json',dict(all_five=aggregate,
         config=protocol,asset_hashes=manifest['assets'],
         full_coverage=dict(questions=n,answers=len(generations),scores=len(scores),grades=len(grades))))
    if readme:
        start,end = '<!-- RESULTS:START -->','<!-- RESULTS:END -->'
        text = readme.read_text(encoding='utf-8')
        before,tail = text.split(start,1)
        _,after = tail.split(end,1)
        content = render_tables(aggregate,seeds,metrics)
        readme.write_text(before+start+'\n'+content+'\n'+end+after,encoding='utf-8')
    print(json.dumps(dict(seeds=seeds,all_five=aggregate),indent=2))


def render_tables(all_five,seeds,metrics):
    columns = [('auroc','.4f'),('tpr_5','.2%'),('fpr_5','.2%'),('pass_at_1','.2%'),('self_ppl','.4f')]
    headings = '| Method | AUROC | TPR at ≤5% FPR | Actual FPR | Pass@1 | Self-PPL |\n|---|---:|---:|---:|---:|---:|'
    content = ['### All five seeds', '', 'Mean ± sample standard deviation across the five generation seeds.', '', headings]
    for method,d in all_five.items():
        values = ['—' if k not in d else f"{d[k]['mean']:{f}} ± {d[k]['sd']:{f}}" for k,f in columns]
        content.append('| '+' | '.join([DISPLAY[method],*values])+' |')
    content += ['', '### Per-seed results', '',
      '| Seed | CodonSeal AUROC | Oracle AUROC | CodonSeal Pass@1 | Oracle Pass@1 | Unmarked Pass@1 |',
      '|---:|---:|---:|---:|---:|---:|']
    by_seed = {(x['seed'],x['method']):x for x in metrics}
    for seed in seeds:
        s,o,u = [by_seed[seed,m] for m in ('CodonSeal','CodonSeal-oracle','unmarked')]
        content.append(f"| {seed} | {s['auroc']:.4f} | {o['auroc']:.4f} | {s['pass_at_1']:.2%} | {o['pass_at_1']:.2%} | {u['pass_at_1']:.2%} |")
    return '\n'.join(content)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('local/run'))
    parser.add_argument('--output',type=Path,default=Path('results'))
    parser.add_argument('--update-readme',action='store_true')
    args = parser.parse_args()
    summarize(args.run,args.output,ROOT/'README.md' if args.update_readme else None)


if __name__=='__main__':
    main()
