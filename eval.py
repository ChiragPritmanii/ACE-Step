import os
import json
from glob import glob

def gen_paths_to_txt(run, step):
    root = "/home/chirag_pritmani24"
    pred_wav = f"{root}/training_logs/{run}_logs/eval_results/step_{step}/pred*wav"
    
    res_path = f"{root}/training_logs/{run}_logs/eval_results/step_{step}/"
    txt_path = f"{root}/training_logs/{run}_logs/eval_results/step_{step}/eval.txt"
        
    os.makedirs(res_path, exist_ok=True)
    wavs = sorted(glob(pred_wav))
    with open(txt_path, "w") as f: 
        for wav in wavs:
            f.write(wav + "\n")

    tup = (txt_path, res_path)
    return tup

def get_seval(run, step):
    root = "/home/chirag_pritmani24"
    res_path = f"{root}/training_logs/{run}_logs/eval_results/step_{step}/result.json"
    seval = json.load(open(res_path, "r"))
    co, mu, mem, cl, nat = 0, 0, 0, 0, 0
    for k, v in seval.items():
        for ki, vi in seval[k].items():
            if ki=="Coherence":
                co+=seval[k][ki]
            elif ki=="Musicality":
                mu+=seval[k][ki]
            elif ki=="Memorability":
                mem+=seval[k][ki]
            elif ki=="Clarity":
                cl+=seval[k][ki]
            elif ki=="Naturalness":
                nat+=seval[k][ki]
        
    co /= len(seval)
    mu /= len(seval)
    mem /= len(seval)
    cl /= len(seval)
    nat /= len(seval)

    print(f"songeval [co, mu, mem, cl, nat]: {[co, mu, mem, cl, nat]}")
    return co, mu, mem, cl, nat