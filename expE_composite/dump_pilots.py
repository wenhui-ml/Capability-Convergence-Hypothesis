import json
S = './scratch'
for name in ['p2_state', 'p2_hyb', 'p2_hybbig']:
    d = json.load(open(f'{S}/{name}.json'))
    ev = d['final_acc']
    print(f"== {d['arch']} ({d['n_params']} params, {d['wall_seconds']}s)")
    for T in [40, 64, 128, 256]:
        print('  T%-3d acc:' % T, {N: round(ev[f'T{T}_N{N}'], 3) for N in [4, 16, 64]},
              ' aux:', {N: round(ev[f'aux_T{T}_N{N}'], 3) for N in [4, 16, 64]})
