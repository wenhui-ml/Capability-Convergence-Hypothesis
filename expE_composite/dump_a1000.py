import glob, json
import numpy as np

d = {}
for f in glob.glob('results/expA/followup/a1_*_n1000.json'):
    j = json.load(open(f))
    d[j['model'].split('/')[-1]] = j['cells']

for cls, (p, m, r) in [('2.8b', ('pythia-2.8b', 'mamba-2.8b-hf', 'rwkv-4-3b-pile')),
                       ('1.4b', ('pythia-1.4b', 'mamba-1.4b-hf', 'rwkv-4-1b5-pile'))]:
    for N in [16, 32, 64]:
        key = f'N{N}_d0.1'
        if p not in d or m not in d:
            continue
        pa, ma = d[p][key]['acc'], d[m][key]['acc']
        se = np.sqrt(pa * (1 - pa) / 1000 + ma * (1 - ma) / 1000)
        line = (f"{cls} N{N}: pythia {pa:.3f} mamba {ma:.3f} "
                f"gap {100*(pa-ma):.1f}pp +-{196*se:.1f} "
                f"CI [{100*(pa-ma)-196*se:.1f}, {100*(pa-ma)+196*se:.1f}]")
        if r in d:
            ra = d[r][key]['acc']
            ser = np.sqrt(pa * (1 - pa) / 1000 + ra * (1 - ra) / 1000)
            line += f" | rwkv {ra:.3f} gap {100*(pa-ra):.1f}pp +-{196*ser:.1f}"
        print(line)
