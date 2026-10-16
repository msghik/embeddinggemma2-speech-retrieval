import os as _os; BASE = _os.environ.get("EG2_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, json, numpy as np, collections
sys.path.insert(0, f"{BASE}/scripts")
from common import *
tag = sys.argv[1] if len(sys.argv) > 1 else "base"
mp = sys.argv[2] if len(sys.argv) > 2 else "google/embeddinggemma-2"
m = load_model(mp)
TPL = 'someone saying the word "{}"'
T = m.encode([TPL.format(w) for w in WORDS], prompt_name="Classification", convert_to_numpy=True)
files, y = split("test"); A = np.load(f"results/{tag}_test_audio.npy")
pred = (A @ T.T).argmax(1)
print(f"text-match test acc: {(pred==y).mean():.4f}")
pc = {WORDS[c]: float((pred[y==c]==c).mean()) for c in range(len(WORDS))}
print("worst classes:", sorted(pc.items(), key=lambda x: x[1])[:8])
print("best classes:", sorted(pc.items(), key=lambda x: -x[1])[:5])
conf = collections.Counter((WORDS[a], WORDS[b]) for a, b in zip(y, pred) if a != b)
print("top confusions (true -> predicted):", conf.most_common(15))
# hard negative per class: most frequent wrong text label on the validation split
vf, vy = split("validation"); V = np.load(f"results/{tag}_validation_audio.npy")
S = V @ T.T
hard = {}
for c in range(len(WORDS)):
    s = S[vy==c].mean(0); s[c] = -9; hard[WORDS[c]] = WORDS[int(s.argmax())]
json.dump({"per_class": pc, "hard_neg": hard}, open(f"results/{tag}_analysis.json", "w"), indent=1)
print("hard negatives:", hard)
# centroid baseline (uses labeled train audio, no training)
trf, ty = split("train")
import os
p = f"results/{tag}_train_audio.npy"
if not os.path.exists(p): np.save(p, embed_audio(m, trf))
Tr = np.load(p)
for k in [1, 10, 100, None]:
    C = []
    rng = np.random.default_rng(0)
    for c in range(len(WORDS)):
        idx = np.where(ty==c)[0]
        if k: idx = rng.choice(idx, k, replace=False)
        v = Tr[idx].mean(0); C.append(v/np.linalg.norm(v))
    C = np.stack(C)
    print(f"centroid {k or 'all'}-shot per class: test acc={((A@C.T).argmax(1)==y).mean():.4f}")
print(f"train clips: {len(trf)}")
