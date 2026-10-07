"""Pixel-structure forensics for every image in ../raw -> ../site/{img/,data.json}."""
import json, os, sys
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__)) + '/..'
RAW, SITE = ROOT + '/raw', ROOT + '/site'
IMG = SITE + '/img'
os.makedirs(IMG, exist_ok=True)


def gblur(x, s):
    r = int(3 * s + .5); k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * s * s)); k /= k.sum()
    p = np.pad(x, ((r, r), (0, 0)), mode='reflect'); y = sum(k[j] * p[j:j + x.shape[0]] for j in range(2 * r + 1))
    p = np.pad(y, ((0, 0), (r, r)), mode='reflect'); return sum(k[j] * p[:, j:j + x.shape[1]] for j in range(2 * r + 1)).astype(np.float32)


def cmap(v):  # v in 0..1 -> magma-like rgb
    t = np.clip(v, 0, 1)
    return (np.stack([np.clip(1.5 * t, 0, 1), np.clip(1.6 * t - .5, 0, 1) ** 1.2, np.clip(.6 * np.sin(np.pi * t) + t ** 3, 0, 1)], -1) * 255).astype(np.uint8)


def analyse(path, base):
    im = Image.open(path)
    alpha_info = None
    if im.mode == 'RGBA':
        al = np.asarray(im)[..., 3]; alpha_info = 'пустая' if al.min() == 255 else 'используется'
    a = np.asarray(im.convert('RGB'), dtype=np.float32)
    H, W, _ = a.shape
    g = (a[..., 0] * .299 + a[..., 1] * .587 + a[..., 2] * .114).astype(np.float32)
    resid = g - gblur(g, 1.2)
    d = {'base': base, 'w': W, 'h': H, 'mb': round(os.path.getsize(path) / 1e6, 1), 'alpha': alpha_info}

    # ---- visual tiles
    t = Image.fromarray(a.astype(np.uint8)); t.thumbnail((900, 900), Image.LANCZOS); t.save(f'{IMG}/{base}_thumb.jpg', quality=86)
    cx, cy = W // 2, int(H * .55)
    Image.fromarray(a[cy - 256:cy + 256, cx - 256:cx + 256].astype(np.uint8)).save(f'{IMG}/{base}_crop.png')
    # macro 32x32 from the most detailed block near the crop centre
    win = resid[cy - 256:cy + 256, cx - 256:cx + 256]
    e = np.abs(win).reshape(16, 32, 16, 32).mean((1, 3)); by, bx = np.unravel_index(np.argsort(e.ravel())[len(e.ravel()) * 3 // 4], e.shape)
    my, mx = cy - 256 + by * 32, cx - 256 + bx * 32
    Image.fromarray(a[my:my + 32, mx:mx + 32].astype(np.uint8)).resize((256, 256), Image.NEAREST).save(f'{IMG}/{base}_macro.png')
    Image.fromarray(np.clip(128 + resid[cy - 128:cy + 128, cx - 128:cx + 128] * 12, 0, 255).astype(np.uint8)).save(f'{IMG}/{base}_resid.png')

    # ---- Welch spectra (residual and luma), 512 tiles, up to 40 tiles
    T = 512; hw = np.outer(np.hanning(T), np.hanning(T)).astype(np.float32)
    Pr = np.zeros((T, T)); Pl = np.zeros((T, T)); n = 0
    ys = list(range(0, H - T + 1, T)); xs = list(range(0, W - T + 1, T))
    pairs = [(y, x) for y in ys for x in xs]
    step = max(1, len(pairs) // 40)
    for y, x in pairs[::step]:
        tr = resid[y:y + T, x:x + T]; tl = g[y:y + T, x:x + T]
        Pr += np.abs(np.fft.fft2((tr - tr.mean()) * hw)) ** 2
        Pl += np.abs(np.fft.fft2((tl - tl.mean()) * hw)) ** 2; n += 1
    Pr = np.fft.fftshift(Pr / n); Pl = np.fft.fftshift(Pl / n)
    L = np.log10(Pr + 1e-6); lo, hi = np.percentile(L, 1), np.percentile(L, 99.95)
    Image.fromarray(cmap((L - lo) / (hi - lo))).resize((256, 256), Image.LANCZOS).save(f'{IMG}/{base}_fft.png')

    c = T // 2
    def Q(y, x): return Pr[y % T, x % T]
    def peak(p):
        k = T // p
        v = np.mean([Q(c, c + k), Q(c, c - k), Q(c + k, c), Q(c - k, c)])
        nb = np.median([Q(c, c + k + o) for o in (-6, -5, -4, -3, 3, 4, 5, 6)] + [Q(c + k + o, c) for o in (-6, -5, -4, -3, 3, 4, 5, 6)])
        return round(float(10 * np.log10(v / nb)), 1)
    d['grid'] = {str(p): peak(p) for p in (2, 3, 4, 8, 16)}
    # off-axis isolated spikes: count residual-spectrum bins > 12 dB above local median (excluding axes)
    Lr = 10 * np.log10(Pr + 1e-9)
    med = np.median(np.stack([np.roll(np.roll(Lr, dy, 0), dx, 1) for dy in (-4, 0, 4) for dx in (-4, 0, 4)]), 0)
    yy, xx = np.indices(Pr.shape) - c; mask = (np.abs(yy) > 6) & (np.abs(xx) > 6) & (np.hypot(yy, xx) > 20)
    d['spikes'] = int(((Lr - med) > 12)[mask].sum())

    rad = np.hypot(*(np.indices((T, T)) - c)).astype(int)
    rs = np.bincount(rad.ravel(), Pl.ravel()) / np.bincount(rad.ravel())
    rs = 10 * np.log10(rs[1:c] + 1e-9); rs = rs - rs[3]
    d['radial'] = [round(float(v), 2) for v in rs]
    band = lambda a0, a1: float(np.mean(rs[int(a0 * T) - 1:int(a1 * T) - 1]))
    d['tail'] = round(band(.40, .49) - band(.20, .25), 1)        # how fast the last octave dies (dB)

    # ---- autocorrelation of residual
    tt = resid[H // 2 - 256:H // 2 + 256, W // 2 - 256:W // 2 + 256]
    F = np.fft.fft2(tt - tt.mean()); ac = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(F) ** 2))); ac /= ac[256, 256]
    acc = ac[256 - 6:256 + 7, 256 - 6:256 + 7]
    Image.fromarray(cmap(np.clip(acc, 0, 1))).resize((130, 130), Image.NEAREST).save(f'{IMG}/{base}_acorr.png')
    d['ac1'] = round(float((ac[256, 257] + ac[257, 256]) / 2), 2)
    d['ac2'] = round(float((ac[256, 258] + ac[258, 256]) / 2), 2)
    d['ac3'] = round(float((ac[256, 259] + ac[259, 256]) / 2), 2)

    # ---- noise vs brightness on flat 8x8 blocks
    B = 8; h8, w8 = H // B * B, W // B * B
    mean = g[:h8, :w8].reshape(H // B, B, W // B, B).mean((1, 3))
    std = resid[:h8, :w8].reshape(H // B, B, W // B, B).std((1, 3))
    gs = gblur(g, 3); gr = np.abs(np.diff(gs, axis=1, append=gs[:, -1:])) + np.abs(np.diff(gs, axis=0, append=gs[-1:]))
    grad = gr[:h8, :w8].reshape(H // B, B, W // B, B).mean((1, 3))
    flat = grad < np.percentile(grad, 20)
    curve = []
    for lo_ in range(0, 256, 16):
        m = flat & (mean >= lo_) & (mean < lo_ + 16)
        curve.append(round(float(np.median(std[m])), 2) if m.sum() > 40 else None)
    d['nvb'] = curve
    d['flat_noise'] = round(float(np.median(std[flat])), 2)

    # ---- channel coupling of fine detail
    rc = [a[..., k] - gblur(a[..., k], 1.2) for k in range(3)]
    sub = (slice(None, None, 3), slice(None, None, 3))
    cc = lambda p, q: round(float(np.corrcoef(p[sub].ravel(), q[sub].ravel())[0, 1]), 2)
    d['crg'], d['cbg'] = cc(rc[0], rc[1]), cc(rc[2], rc[1])

    # ---- histogram / clipping / blockiness
    hist = np.bincount(np.clip(g, 0, 255).astype(np.uint8).ravel(), minlength=256)
    d['hist'] = [int(v) for v in hist]
    px = a.reshape(-1, 3)
    d['clipW'] = round(float((px.min(1) >= 254.5).mean() * 100), 2)   # all three channels at 255
    d['clipB'] = round(float((px.max(1) <= .5).mean() * 100), 2)
    dc = np.abs(np.diff(g, axis=1)).mean(0); d['blk8'] = round(float(dc[7::8].mean() / np.delete(dc, np.s_[7::8]).mean()), 3)
    return d


if __name__ == '__main__':
    out = {}
    if os.path.exists(SITE + '/data.json'):
        out = {r['base']: r for r in json.load(open(SITE + '/data.json'))['images']}
    for f in sorted(os.listdir(RAW)):
        base = f.rsplit('.', 1)[0]
        if base in out and '--force' not in sys.argv: continue
        out[base] = analyse(f'{RAW}/{f}', base)
        r = out[base]
        print(f"{base:34s} {r['w']}x{r['h']} ac {r['ac1']}/{r['ac2']} grid {r['grid']} spikes {r['spikes']} tail {r['tail']} ch {r['crg']}/{r['cbg']} flat {r['flat_noise']} clipW {r['clipW']}", flush=True)
        json.dump({'images': list(out.values())}, open(SITE + '/data.json', 'w'))
