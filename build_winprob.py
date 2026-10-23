"""訓練即時勝率模型 → winprob_model.json（面板讀取）。

步驟：
1. 對 history.json 的每局，每 3 分鐘抽一個場上狀態（經濟/擊殺/塔/龍/巴龍/兵營），
   並用二分搜尋找到最後一個 frame 判定勝方。結果快取在 winprob_samples.json。
2. 依時間順序計算各隊 Elo（賽前實力）。
3. 純 Python 邏輯斯迴歸；以最近 15% 的對局做驗證。

用法：先跑 build_history.py，再跑 python build_winprob.py
     python build_winprob.py --train-only  只用已快取的資料重新訓練
"""
import json
import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from build_history import get

FEED = 'https://feed.lolesports.com/livestats/v1/window/'
DIR = Path(__file__).parent
SAMPLES = DIR / 'winprob_samples.json'
OUT = DIR / 'winprob_model.json'
SAMPLE_EVERY = 180  # 秒
ELO_K, ELO_INIT = 32, 1500

FEATURES = ['bias', 'gold', 'gold_x_t', 'kills', 'towers', 'towers_x_t', 'dragons', 'soul', 'elder',
            'barons', 'inhibs', 'elo', 'elo_x_early']


def ms(ts):
    return datetime.fromisoformat(ts.replace('Z', '+00:00'))


def iso(dt):
    dt = dt.replace(microsecond=0, second=dt.second // 10 * 10)
    return dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')


def window(game_id, at=None):
    w = get(FEED + game_id + (f'?startingTime={iso(at)}' if at else ''))
    return w['frames'][-1] if w and w.get('frames') else None


def state(f):
    out = {}
    for side in ('blue', 'red'):
        t = f[side + 'Team']
        out[side] = [t['totalGold'], t['totalKills'], t['towers'], t['dragons'], t['barons'], t['inhibitors']]
    return out


def collect(game_id):
    first = get(FEED + game_id)
    if not first or not first.get('frames'):
        return None
    t0 = ms(first['frames'][0]['rfc460Timestamp'])
    samples, last_ok, k = [], t0, 1
    while k < 40:  # 最長 2 小時
        at = t0 + timedelta(seconds=k * SAMPLE_EVERY)
        f = window(game_id, at)
        if f and f['gameState'] != 'finished':
            samples.append({'t': (ms(f['rfc460Timestamp']) - t0).total_seconds(), **state(f)})
            last_ok, k = at, k + 1
            continue
        final = f if f else None
        if not final:  # 二分搜尋最後一個有資料的時間點
            lo, hi = last_ok, at
            while (hi - lo).total_seconds() > 10:
                mid = lo + (hi - lo) / 2
                g = window(game_id, mid)
                if g:
                    lo, final = mid, g
                else:
                    hi = mid
            final = final or window(game_id, lo)
        if not final:
            return None
        s = state(final)
        # 勝方：推塔多者（主堡砲塔/主堡算在內）；平手看兵營、再看經濟
        key = lambda side: (s[side][2], s[side][5], s[side][0])
        return {'samples': samples, 'blueWin': key('blue') > key('red'),
                'length': (ms(final['rfc460Timestamp']) - t0).total_seconds()}
    return None


def features(s, t, elo_diff):
    b, r = s['blue'], s['red']
    tm = t / 600  # 以 10 分鐘為單位
    soul = (len(b[3]) >= 4) - (len(r[3]) >= 4)
    elder = b[3].count('elder') - r[3].count('elder')
    return [1, (b[0] - r[0]) / 1000, (b[0] - r[0]) / 1000 * tm, b[1] - r[1], b[2] - r[2], (b[2] - r[2]) * tm,
            len(b[3]) - len(r[3]), soul, elder, b[4] - r[4], b[5] - r[5],
            elo_diff / 100, elo_diff / 100 * max(0, 1 - tm / 3)]


def sigmoid(z):
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


def fit(X, y, l2=1e-2, iters=25):
    """邏輯斯迴歸（牛頓法 / IRLS，L2 正則化）。"""
    d = len(X[0])
    w = [0.0] * d
    for _ in range(iters):
        g = [l2 * w[j] * (j > 0) for j in range(d)]
        Hm = [[l2 * (i == j and i > 0) for j in range(d)] for i in range(d)]
        for x, yy in zip(X, y):
            p = sigmoid(sum(a * b for a, b in zip(w, x)))
            e, v = p - yy, p * (1 - p)
            for i in range(d):
                g[i] += e * x[i]
                vi = v * x[i]
                row = Hm[i]
                for j in range(i, d):
                    row[j] += vi * x[j]
        for i in range(d):
            for j in range(i):
                Hm[i][j] = Hm[j][i]
        step = solve(Hm, g)
        w = [wi - si for wi, si in zip(w, step)]
        if max(abs(si) for si in step) < 1e-6:
            break
    return w


def solve(A, b):
    """高斯消去法解 A x = b。"""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        if abs(M[c][c]) < 1e-12:
            M[c][c] = 1e-12
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def evaluate(w, X, y):
    ll = acc = 0
    for x, yy in zip(X, y):
        p = min(max(sigmoid(sum(a * b for a, b in zip(w, x))), 1e-6), 1 - 1e-6)
        ll -= yy * math.log(p) + (1 - yy) * math.log(1 - p)
        acc += (p > 0.5) == yy
    return ll / len(X), acc / len(X)


def main():
    games = json.loads((DIR / 'history.json').read_text(encoding='utf-8'))['games']
    cache = json.loads(SAMPLES.read_text(encoding='utf-8')) if SAMPLES.exists() else {}
    todo = [] if '--train-only' in sys.argv else [g['id'] for g in games if g['id'] not in cache]
    print(f'需抓取 {len(todo)} 局的場上狀態', flush=True)
    with ThreadPoolExecutor(16) as pool:
        for i, (gid, r) in enumerate(zip(todo, pool.map(lambda x: _safe(collect, x), todo)), 1):
            cache[gid] = r
            if i % 100 == 0:
                print(f'  {i}/{len(todo)}', flush=True)
                SAMPLES.write_text(json.dumps(cache), encoding='utf-8')
    if todo:
        SAMPLES.write_text(json.dumps(cache), encoding='utf-8')

    # Elo（依時間順序，用賽前分數當特徵，避免資訊外洩）
    elo, rows = {}, []
    for g in sorted(games, key=lambda g: g['date']):
        c = cache.get(g['id'])
        if not c:
            continue
        bt, rt = g['blue']['teamId'], g['red']['teamId']
        eb, er = elo.get(bt, ELO_INIT), elo.get(rt, ELO_INIT)
        rows.append((g, c, eb - er))
        exp = 1 / (1 + 10 ** ((er - eb) / 400))
        delta = ELO_K * (c['blueWin'] - exp)
        elo[bt], elo[rt] = eb + delta, er - delta

    split = int(len(rows) * 0.85)

    def xy(rs):
        X, y, T = [], [], []
        empty = {'blue': [0, 0, 0, [], 0, 0], 'red': [0, 0, 0, [], 0, 0], 't': 0}  # 開局（只有 Elo）
        for g, c, ed in rs:
            for s in [empty] + c['samples']:
                X.append(features(s, s['t'], ed))
                y.append(1 if c['blueWin'] else 0)
                T.append(s['t'])
        return X, y, T

    Xtr, ytr, _ = xy(rows[:split])
    Xte, yte, Tte = xy(rows[split:])
    print(f'訓練 {len(Xtr)} 筆 / 驗證 {len(Xte)} 筆（{len(rows)} 局，藍方勝率 {sum(c["blueWin"] for _, c, _ in rows) / len(rows):.1%}）', flush=True)
    w = fit(Xtr, ytr)
    ll, acc = evaluate(w, Xte, yte)
    print(f'驗證 log-loss {ll:.3f}，準確率 {acc:.1%}')
    for lo, hi in [(0, 1), (500, 700), (1100, 1300), (1700, 1900)]:
        idx = [i for i, t in enumerate(Tte) if lo <= t < hi]
        if idx:
            print(f'  {lo // 60:>2}–{hi // 60} 分：準確率 {evaluate(w, [Xte[i] for i in idx], [yte[i] for i in idx])[1]:.1%}（{len(idx)} 筆）')
    w = fit(Xtr + Xte, ytr + yte)  # 用全部資料重訓最終模型

    teams = {}
    for g in games:
        for side in ('blue', 'red'):
            teams[g[side]['teamId']] = g[side]['code']
    OUT.write_text(json.dumps({
        'features': FEATURES, 'weights': [round(v, 5) for v in w], 'games': len(rows),
        'validation': {'logloss': round(ll, 4), 'accuracy': round(acc, 4)},
        'elo': {k: round(v, 1) for k, v in elo.items()}, 'teams': teams, 'eloInit': ELO_INIT,
    }, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'完成 → {OUT}')
    for code, tid in sorted(((teams.get(k, '?'), k) for k in elo), key=lambda x: -elo[x[1]])[:12]:
        print(f'  {code:>5} {elo[tid]:.0f}')


def _safe(fn, x):
    try:
        return fn(x)
    except Exception:
        return None


if __name__ == '__main__':
    random.seed(0)
    main()
