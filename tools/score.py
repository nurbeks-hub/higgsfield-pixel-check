"""Turn per-image metrics into penalties, a 0-100 pixel score and per-model aggregates."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
S = HERE + '/../site/data.json'
D = json.load(open(S))

NAMES = {
    'flux_3_image': 'FLUX 3', 'flux_2': 'FLUX.2 Max', 'flux_kontext': 'Flux Kontext', 'gpt_image_2': 'GPT Image 2',
    'gpt_image_2_5': 'GPT Image 2.5 · flare', 'gpt_image_2_5-sunburst': 'GPT Image 2.5 · sunburst', 'grok_image': 'Grok Image',
    'grok_image_2_0': 'Grok Image 2.0', 'text2image_soul_v2': 'Soul 2.0', 'ideogram_4_5': 'Ideogram 4.5', 'kling_omni_image': 'Kling O1 Image',
    'nano_banana': 'Nano Banana', 'nano_banana_flash': 'Nano Banana 2', 'nano_banana_2_lite': 'Nano Banana 2 Lite',
    'nano_banana_2_1': 'Nano Banana 2.1', 'nano_banana_pro': 'Nano Banana Pro', 'openai_hazel': 'OpenAI Hazel', 'recraft_v4_1': 'Recraft V4.1',
    'seedream_v4_5': 'Seedream 4.5', 'seedream_5_0_flash': 'Seedream 5.0 Flash', 'seedream_v5_lite': 'Seedream 5.0 Lite',
    'seedream_v5_pro': 'Seedream 5.0 Pro', 'soul_cinematic': 'Soul Cinematic', 'soul_location': 'Soul Location', 'z_image': 'Z Image',
    'cinematic_studio_2_5': 'Cinematic Studio 2.5', 'cinematic_studio_soul_location': 'Cinematic Studio Soul Location',
}
ARGS = {}
for line in open(HERE + '/configs.tsv'):
    k, a = line.rstrip('\n').split('\t'); ARGS[k.replace('@', '-')] = a.replace('--', '').replace('_', ' ')

clip = lambda v: max(0.0, min(1.0, v))
W = {'up': .20, 'res': .10, 'grid': .20, 'alias': .10, 'noise': .15, 'clip': .10, 'chan': .15}


def penalties(r):
    g = max(r['grid'].values())
    fn = r['flat_noise']
    p = {
        'up': clip(max((r['eff_scale'] - 1) / 1.5, (r['ac2'] - .08) / .30)),
        'res': clip((2.5 - r['eff']) / 2),
        'grid': clip((g - 6) / 24),
        'alias': clip(r['spikes'] / 25),
        'noise': clip((.4 - fn) / .4) if fn < .4 else clip((fn - 4) / 5),
        'clip': clip(r['clipW'] / 5),
        'chan': clip(((r['crg'] + r['cbg']) / 2 - .80) / .19),
    }
    return {k: round(v, 3) for k, v in p.items()}


def flags(r, p):
    out = []
    if p['res'] >= .5: out.append(f"мало пикселей: {r['eff']} Мп")
    if p['up'] >= .5: out.append(f"апскейл: реально ≈{r['eff']} Мп из {round(r['w'] * r['h'] / 1e6, 1)}")
    elif p['up'] >= .2: out.append('частично растянута')
    g = max(r['grid'], key=lambda k: r['grid'][k])
    if r['grid'][g] >= 12: out.append(f"сетка с шагом {g} px (+{r['grid'][g]} dB)")
    elif r['grid'][g] >= 6: out.append(f"слабая сетка {g} px (+{r['grid'][g]} dB)")
    if r['spikes'] >= 12: out.append(f"алиасинг: {r['spikes']} точечных пиков")
    if r['flat_noise'] < .4: out.append('пластик: шума на ровном почти нет')
    if r['flat_noise'] > 4: out.append(f"очень сильное зерно (σ {r['flat_noise']})")
    if r['clipW'] >= 1: out.append(f"{r['clipW']}% пикселей выбито в белый")
    if (r['crg'] + r['cbg']) / 2 < .85: out.append('цветное зерно, каналы независимы')
    if r.get('alpha') == 'пустая': out.append('пустой альфа-канал')
    return out


models = {}
for r in D['images']:
    key = r['base'].rsplit('_', 1)[0]
    r['model'] = key
    r['pen'] = penalties(r)
    r['score'] = round(100 * (1 - sum(W[k] * v for k, v in r['pen'].items())), 1)
    r['flags'] = flags(r, r['pen'])
    models.setdefault(key, []).append(r)

out = []
for key, rs in models.items():
    rs.sort(key=lambda r: r['base'])
    pen = {k: round(sum(r['pen'][k] for r in rs) / len(rs), 3) for k in W}
    out.append({
        'key': key, 'name': NAMES.get(key, key), 'args': ARGS.get(key, ''), 'takes': [r['base'] for r in rs],
        'score': round(sum(r['score'] for r in rs) / len(rs), 1), 'spread': round(abs(rs[0]['score'] - rs[-1]['score']), 1),
        'pen': pen, 'w': rs[0]['w'], 'h': rs[0]['h'], 'eff': round(sum(r['eff'] for r in rs) / len(rs), 2),
    })
out.sort(key=lambda m: -m['score'])
D['models'] = out
D['weights'] = W
D['prompt'] = open(HERE + '/prompt.txt').read().strip()
json.dump(D, open(S, 'w'))
for i, m in enumerate(out, 1):
    print(f"{i:2d} {m['score']:5.1f} ±{m['spread']:<4} {m['name']:32s} {m['w']}x{m['h']} eff {m['eff']} {m['pen']}")
