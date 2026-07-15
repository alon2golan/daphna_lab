"""Faithful offline replica of daphna_lab ProbCover + 1-NN (framework ii).

Reproduces train_al.py: seed -> makeLUVSets split -> ProbCover greedy select
-> 1-NN on L2-normalized features, eval on the 10k CIFAR-100 test set.
Lets us toggle ONE factor at a time to find the root cause of 38.1 vs paper ~34-35.
"""
import numpy as np, torch, pickle, argparse

EMB = {
 "itai":   ("/cs/labs/daphna/itai.david/representations_bank/cifar-100_simclr/cifar100_simclr_train.npy",
            "/cs/labs/daphna/itai.david/representations_bank/cifar-100_simclr/cifar100_simclr_test.npy"),
 "typiclust": ("/cs/labs/daphna/alongol/data/embeddings/cifar-100/downloaded_from_typiclust/features_seed1.npy",
            None),  # no matching 10k test embedding
 "alon":   ("/cs/labs/daphna/alongol/data/embeddings/cifar-100/alon_20260217/features_seed1.npy",
            "/cs/labs/daphna/alongol/data/embeddings/cifar-100/alon_20260217/test_features_seed1.npy"),
}
def labels(split):
    p=f"/cs/labs/daphna/alongol/projects/daphna_lab/data/cifar-100-python/{split}"
    with open(p,'rb') as f: d=pickle.load(f,encoding='latin1')
    return np.array(d['fine_labels'])

def l2(x): return x/np.linalg.norm(x,axis=1,keepdims=True)

def make_uset(seed, n=50000, val_ratio=0.1, init_l_ratio=0.0):
    # exact replica of makeLUVSets
    torch.manual_seed(seed); np.random.seed(seed)
    all_idx=list(range(n)); np.random.shuffle(all_idx)
    train_splitIdx=int(init_l_ratio*n); val_splitIdx=int((1-val_ratio)*n)
    uSet=np.array(all_idx[train_splitIdx:val_splitIdx])
    return uSet

def probcover_select(feats_norm, uSet, delta, num, mode="argmax", rng=None):
    rel=uSet
    rf=torch.tensor(feats_norm[rel]).float()
    bs=500; xs=[];ys=[]
    for i in range(int(np.ceil(len(rf)/bs))):
        cur=rf[i*bs:(i+1)*bs]
        dist=torch.cdist(cur,rf)
        mask=dist<delta
        x,y=mask.nonzero(as_tuple=False).T
        xs.append((x+bs*i).numpy()); ys.append(y.numpy())
    x=np.concatenate(xs); y=np.concatenate(ys)
    N=len(rel)
    covered=np.array([],dtype=int)
    keep=np.ones(len(y),dtype=bool)  # edges still active
    selected=[]
    for it in range(num):
        deg=np.bincount(x[keep],minlength=N)
        if mode=="argmax":
            cur=int(deg.argmax())
        elif mode=="rand_top5":
            top5=deg.argsort()[::-1][:5]
            cur=int(rng.choice(top5))
        newcov=y[keep & (x==cur)]
        drop=np.isin(y,newcov)
        keep=keep & (~drop)
        covered=np.concatenate([covered,newcov])
        selected.append(cur)
    return rel[np.array(selected)]

def run(emb, seed, delta=0.75, mode="argmax", norm_sel=True, norm_1nn=True):
    tr_p,te_p=EMB[emb]
    tr=np.load(tr_p); te=np.load(te_p) if te_p else None
    ytr=labels("train"); yte=labels("test")
    feats_sel = l2(tr) if norm_sel else tr
    uSet=make_uset(seed)
    rng=np.random.RandomState(seed)
    sel=probcover_select(feats_sel, uSet, delta, 100, mode, rng)
    # 1-NN
    from sklearn.neighbors import KNeighborsClassifier
    Xtr = l2(tr) if norm_1nn else tr
    Xte = l2(te) if norm_1nn else te
    knn=KNeighborsClassifier(n_neighbors=1).fit(Xtr[sel], ytr[sel])
    acc=100.*(knn.predict(Xte)==yte).mean()
    return acc, sel

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--emb",default="itai"); ap.add_argument("--seed",type=int,default=1)
    ap.add_argument("--delta",type=float,default=0.75); ap.add_argument("--mode",default="argmax")
    a=ap.parse_args()
    acc,sel=run(a.emb,a.seed,a.delta,a.mode)
    print(f"emb={a.emb} seed={a.seed} delta={a.delta} mode={a.mode}  ACC={acc:.2f}")
