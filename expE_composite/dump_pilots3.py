import json
S = './scratch'
for name in ['p3_state', 'p3_hyb', 'p3_hybbig']:
    try:
        d = json.load(open(f'{S}/{name}.json'))
    except FileNotFoundError:
        print(f'== {name}: MISSING')
        continue
    ev = d['final_acc']
    print(f"== {d['arch']} ({d['n_params']} params, {d['wall_seconds']}s, tied)")
    for T in [40, 64, 128, 256]:
        print('  T%-3d acc:' % T, {N: round(ev[f'T{T}_N{N}'], 3) for N in [4, 16, 64]},
              ' aux:', {N: round(ev[f'aux_T{T}_N{N}'], 3) for N in [4, 16, 64]})
