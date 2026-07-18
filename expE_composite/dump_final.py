import glob, json
for f in sorted(glob.glob('results/expE/e_*.json')):
    d = json.load(open(f))
    ev = d['final_acc']
    print(f"{d['arch']:20s} s{d['seed']}  T40: " +
          str({N: round(ev[f'T40_N{N}'], 3) for N in [4, 16, 64]}) +
          "  T256: " + str({N: round(ev[f'T256_N{N}'], 3) for N in [4, 16, 64]}))
