"""Raw-score empirical ROC and paired-question intervals."""
import math
import numpy as np


def roc(negative,positive):
    scores=np.r_[negative,positive];labels=np.r_[np.zeros(len(negative)),np.ones(len(positive))]
    order=np.argsort(-scores,kind='stable');scores=scores[order];labels=labels[order]
    ends=np.r_[np.flatnonzero(np.diff(scores)!=0),len(scores)-1]
    tp=np.cumsum(labels)[ends];fp=ends+1-tp
    fpr=np.r_[0.,fp/len(negative)];tpr=np.r_[0.,tp/len(positive)]
    ap=float(np.sum(np.diff(tpr)*tp/(ends+1)))
    return dict(fpr=fpr.tolist(),tpr=tpr.tolist(),auc=float(np.trapezoid(tpr,fpr)),average_precision=ap)

def wilson(k,n):
    z=1.95996398454;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-radius),min(1.,center+radius)]

def compare(negative,positive,calibration):
    n=np.asarray(negative);p=np.asarray(positive);curve=roc(n,p)
    rng=np.random.default_rng(87);boot=[]
    for _ in range(1500):
        idx=rng.integers(len(n),size=len(n));boot.append(roc(n[idx],p[idx])['auc'])
    if not len(calibration):
        thresholds=np.r_[np.nextafter(max(np.max(n),np.max(p)),np.inf),np.unique(np.r_[n,p])[::-1]]
        points={};chosen=None
        for alpha in [0.,.001,.01,.05,.10]:
            best=(-1.,-1.)
            for value in thresholds:
                f=float(np.mean(n>=value));t=float(np.mean(p>=value))
                if f<=alpha and (t,-f)>best:best=(t,-f);threshold=float(value)
            tp=int(np.sum(p>=threshold));fp=int(np.sum(n>=threshold))
            points[str(alpha)]=dict(tpr=tp/len(p),fpr=fp/len(n),tp=tp,fp=fp,negative_n=len(n),threshold=threshold)
            if alpha==.05:chosen=points[str(alpha)]
        tp,fp=chosen['tp'],chosen['fp'];precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/len(p)
        return dict(auroc=curve['auc'],average_precision=curve['average_precision'],roc=curve,
            paired_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),threshold=chosen['threshold'],
            threshold_operator='>=',threshold_source='same-evaluation empirical ROC, not deployment calibration',
            operating_points=points,nominal_fpr=.05,tpr=recall,fpr=fp/len(n),
            tpr_wilson_95=wilson(tp,len(p)),fpr_wilson_95=wilson(fp,len(n)),tp=tp,fp=fp,tn=len(n)-fp,fn=len(p)-tp,
            precision=precision,f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
            balanced_accuracy=.5*(recall+1-fp/len(n)),test_pairs=len(n),calibration_negatives=0)
    k=math.ceil((len(calibration)+1)*.95)
    if k>len(calibration):raise ValueError('Insufficient calibration negatives for a finite 5% threshold')
    threshold=float(sorted(calibration)[k-1]);tp=int(np.sum(p>threshold));fp=int(np.sum(n>threshold))
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/len(p)
    return dict(auroc=curve['auc'],average_precision=curve['average_precision'],roc=curve,
        paired_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),threshold=threshold,
        nominal_fpr=.05,tpr=recall,fpr=fp/len(n),tpr_wilson_95=wilson(tp,len(p)),fpr_wilson_95=wilson(fp,len(n)),
        tp=tp,fp=fp,tn=len(n)-fp,fn=len(p)-tp,precision=precision,
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
        balanced_accuracy=.5*(recall+1-fp/len(n)),test_pairs=len(n),calibration_negatives=len(calibration))
