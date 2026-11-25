"""Generate one answer or score a received answer with a released detector."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['generate','detect'])
    parser.add_argument('--method',choices=['CodonSeal','CodonSeal-oracle','unmarked'],default='CodonSeal')
    parser.add_argument('--prompt',type=Path,help='UTF-8 file containing the exact user prompt.')
    parser.add_argument('--answer',type=Path,help='UTF-8 file containing the received answer body.')
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--key-file',type=Path,help='External binary key. Uses the public experiment key by default.')
    parser.add_argument('--local',type=Path,default=Path('local'))
    args = parser.parse_args()
    if (args.action=='generate' or args.method=='CodonSeal-oracle') and args.prompt is None:
        parser.error('--prompt is required for generation and for the oracle detector')
    if args.action=='detect' and (args.answer is None or args.method=='unmarked'):
        parser.error('detection requires --answer and a watermarked method')

    import mlx.core as mx
    from codonseal.watermark import BodyDetector,ReleaseBackend,Tiny
    protocol = json.loads((ROOT/'configs/protocol.json').read_text())
    key = args.key_file.read_bytes() if args.key_file else hashlib.sha256(protocol['public_key_label'].encode()).digest()
    tokenizer_dir = args.local/'model'
    detector = None
    if args.action=='detect' and args.method=='CodonSeal':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,local_files_only=True)
    else:
        backend = ReleaseBackend(tokenizer_dir,key)
        tokenizer = backend.tokenizer
    if args.method=='CodonSeal':
        geometry = mx.load(str(args.local/'wpn/geometry.safetensors'))
        net = Tiny(geometry['bank'],2)
        net.load_weights(str(args.local/'wpn/tiny2/best.safetensors'))
        net.eval()
        detector = BodyDetector(net,tokenizer,geometry,key)
    if args.action=='detect':
        answer = args.answer.read_text(encoding='utf-8')
        if detector is not None:
            result = detector.detect_text(answer)
            print(json.dumps(dict(score=result['scores']['sqrt_count'],candidate_keys=result['events'])))
        else:
            prompt = args.prompt.read_text(encoding='utf-8')
            result = backend.replay_qa(dict(text=answer,prompt=prompt,ids=[]),False)
            print(json.dumps(dict(score=result['score'],candidate_keys=result['events'])))
        return
    prompt = args.prompt.read_text(encoding='utf-8')
    row = dict(id='demo',prompt=prompt)
    if detector is not None:
        record = backend.sync_generate([row],detector,False,args.seed)[0]
    else:
        variant = 'baseline' if args.method=='unmarked' else 'full'
        record = backend.generate([(row,variant)],protocol['max_answer_tokens'],args.seed)[0]
    print(record['text'])


if __name__=='__main__':
    main()
