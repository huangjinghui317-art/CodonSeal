"""The frozen two-token, sixteen-slot Open-R1 CodonSeal architecture."""
import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from .backend import MLXBackend
from .core import LockState, color_table, color_features, polarity, pair_seed, probabilities, support
from .util import chunks


class OracleBackend(MLXBackend):
    """Original semantic weight of 2.0 with the prompt-conditioned detector."""
    def features(self, *args, **kwargs):
        color, semantic = super().features(*args, **kwargs)
        return color, 2.0 * semantic

class StateNet(nn.Module):
    def __init__(self,bank):
        super().__init__();self.lookup=nn.Embedding(bank.shape[0],64);self.lookup.weight=bank;self.lookup.freeze()
        self.project=nn.Linear(64,128);self.position=nn.Embedding(16,128)
        self.layers=[nn.TransformerEncoderLayer(128,4,256,.1,norm_first=True) for _ in range(2)]
        self.norm=nn.LayerNorm(128);self.classes=nn.Linear(128,3);self.richness=nn.Linear(128,1)
        self.forecast=nn.Linear(128,1536);self.forecast.weight=self.forecast.weight*.01;self.forecast.bias=mx.zeros(1536)
    def __call__(self,ids):
        h=self.project(self.lookup(ids))+self.position(mx.arange(16))
        for layer in self.layers:h=layer(h,mask=None)
        h=self.norm(h[:,-1]);return self.classes(h),self.richness(h)[:,0],self.forecast(h)

def windows(ids):
    return np.array([[151643]*(16-min(t,16))+ids[max(0,t-16):t] for t in range(len(ids))],dtype=np.int32)

class BodyDetector:
    """No reference to an LLM or original prompt; inputs are received text only."""
    def __init__(self,net,tokenizer,geometry,key):
        self.net=net;self.tokenizer=tokenizer;self.units=geometry['units'];self.mean=geometry['mean']
        self.key=key;self.colors=color_table(key,len(self.units));self.special=set(tokenizer.all_special_ids)|{151643,151645}
    def gate(self,contexts):
        logits,_,forecast=self.net(mx.array(contexts));forecast=forecast/mx.maximum(mx.linalg.norm(forecast,axis=-1,keepdims=True),1.);mx.eval(logits,forecast)
        logits=np.array(logits);classes=np.argmax(np.round(logits*32)/32,axis=-1)
        return np.array([1,3,4])[classes],probabilities(logits),forecast
    def bias(self,ids,locks,reference):
        """Same student-reference codon feature at generation and detection.
        Semantic centering uses the body-only predicted next embedding, scale0.08.
        """
        ids=np.asarray(ids);raw=color_features(self.colors,ids,locks)
        frequencies=np.array([np.bincount(self.colors[list(lock)],minlength=3)/len(lock) for lock in locks])
        scale=np.maximum(np.sqrt(np.mean((frequencies-1/3)**2,axis=1)),.08)
        color=np.clip((raw+1/3)/scale[:,None],-2,2)
        k=max(map(len,locks));index=mx.array([list(lock)+[0]*(k-len(lock)) for lock in locks])
        mask=mx.array([[1.]*len(lock)+[0.]*(k-len(lock)) for lock in locks])
        centers=(self.units[index].astype(mx.float32)*mask[:,:,None]).sum(1)
        centers=centers/mx.maximum(mx.linalg.norm(centers,axis=-1,keepdims=True),1e-8)
        vec=self.units[mx.array(ids)].astype(mx.float32)
        signs=np.array([polarity(self.key,lock) for lock in locks])
        cosine=np.array((vec*centers[:,None,:]).sum(-1));mean=np.array((centers*reference).sum(-1))
        semantic=np.clip((cosine-mean[:,None])*signs[:,None]/.08,-2,2)
        return (color+2*semantic)*(~np.isin(ids,list(self.special)))
    def detect_text(self,text):
        ids=self.tokenizer.encode(text,add_special_tokens=False)
        counts=[];probs=[];forecasts=[]
        for batch in chunks(windows(ids),128):
            c,p,f=self.gate(batch);counts.extend(map(int,c));probs.extend(p.tolist());forecasts.append(f)
        state=LockState();locks=[];positions=[];confidence=[]
        for t,(count,token) in enumerate(zip(counts,ids)):
            lock=state.before(count)
            if lock:
                end=state.sealed_at;lo=max(0,end-len(lock));weight=float(probs[t][2]*np.prod(np.array(probs)[lo:end,0]))
                locks.append(lock);positions.append(t);confidence.append(weight)
            state.after(count,token)
        values=self.bias(np.array([[ids[t]] for t in positions]),locks,mx.concatenate(forecasts)[mx.array(positions)])[:,0].tolist() if locks else []
        n=len(values);total=float(sum(values));scan=0.
        for end in range(len(ids)):
            vv=[v for p,v in zip(positions,values) if end-31<=p<=end]
            if vv:scan=max(scan,float(sum(vv)/math.sqrt(len(vv))))
        c=np.array(confidence);v=np.array(values)
        return dict(scores=dict(sum=total,sqrt_count=total/math.sqrt(max(1,n)),scan32=scan,
            confidence=float((v*c).sum()/math.sqrt(max(float((c*c).sum()),1e-10)))),
            events=n,key_positions=positions,increments=values,counts=counts,confidence=confidence,
            token_hash=__import__('hashlib').sha256(np.asarray(ids,dtype='<i4').tobytes()).hexdigest())


class Tiny(StateNet):
    def __init__(self,bank,width):super().__init__(bank);self.width=width
    def __call__(self,x):
        if self.width<16:x=mx.concatenate([mx.full((len(x),16-self.width),151643,dtype=mx.int32),x[:,-self.width:]],axis=-1)
        return super().__call__(x)


class ReleaseBackend(OracleBackend):
    def sync_generate(self,rows,detector,safe,seed):
        inputs,cache=self.cache([self.prompt(q['prompt']) for q in rows]);rng=[np.random.default_rng(pair_seed(seed,q['id'])) for q in rows]
        states=[LockState() for _ in rows];done=np.zeros(len(rows),bool);gs=[dict(ids=[],trace=[],nll=[],finish='length') for _ in rows]
        for _ in range(256):
            ids,raw,output=self.top_candidates(inputs,cache);actual=support(raw)
            x=np.array([[151643]*(16-min(len(g['ids']),16))+g['ids'][-16:] for g in gs],np.int32);counts,_,ref=detector.gate(x)
            locks=[s.before(int(c)) if not done[i] else () for i,(s,c) in enumerate(zip(states,counts))]
            active=[i for i,lock in enumerate(locks) if lock and (not safe or actual[i]>=4)];logits=raw/.8
            if active:logits[active]+=detector.bias(ids[active],[locks[i] for i in active],ref[mx.array(active)])
            sampled=[int(ids[i,min(63,np.searchsorted(p.cumsum(),rng[i].random()))]) for i,p in enumerate(probabilities(logits))]
            losses=np.array(mx.logsumexp(output,axis=-1)-mx.take_along_axis(output,mx.array(sampled)[:,None],axis=-1)[:,0])
            for i,t in enumerate(sampled):
                if done[i]:continue
                if t in self.stops:done[i]=True;gs[i]['finish']='eos';continue
                gs[i]['ids'].append(t);gs[i]['nll'].append(float(losses[i]));gs[i]['trace'].append(dict(count=int(counts[i]),actual_count=int(actual[i]),lock=list(locks[i]),embedded=i in active));states[i].after(int(counts[i]),t)
            if done.all():break
            inputs=[[t] for t in sampled]
        for g in gs:g['text']=self.tokenizer.decode(g['ids'])
        return gs
