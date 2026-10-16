import os as _os; BASE = _os.environ.get("EG2_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import torch, soundfile as sf, numpy as np
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("google/embeddinggemma-2", model_kwargs={"torch_dtype": torch.bfloat16},
                        config_kwargs={"vision_config": None})
print(m); print(list(m.prompts.keys()))
root=f"{BASE}/data/sc/"
import os
files=[w+"/"+sorted(os.listdir(root+w))[0] for w in ["yes","no","stop","left"]]
for f in files:
    a,sr=sf.read(root+f); print(f, sr, a.shape)
labels=["yes","no","stop","go","left","right"]
t=m.encode(labels, prompt_name="Classification")
for try_ in ["path","dict","array"]:
    try:
        if try_=="path": x=[{"audio": root+f} for f in files]
        elif try_=="dict": x=[{"audio": {"array": sf.read(root+f)[0].astype(np.float32), "sampling_rate":16000}} for f in files]
        else: x=[{"audio": sf.read(root+f)[0].astype(np.float32)} for f in files]
        e=m.encode(x)
        print(try_, e.shape); print(m.similarity(e,t))
    except Exception as ex: print(try_, "FAIL", repr(ex)[:300])
