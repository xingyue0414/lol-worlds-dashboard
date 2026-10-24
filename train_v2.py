"""勝率模型 v2：用 collect_v2.py 的資料訓練 LightGBM，並和目前網站上的模型在同一批近期對局上比較。

步驟：
1. 勝方標籤：最後一刻推塔數判定，再用官方系列賽比分驗證；比分對不上的系列賽整組排除
2. 依時間順序計算 Elo（賽前實力）
3. 最近 15% 的對局當驗證集，比較：目前模型(v1) / 邏輯斯迴歸(新特徵) / LightGBM(新特徵)
4. 輸出 winprob_model.json（網站讀取；LightGBM 以樹結構 JSON 匯出，由 winprob.js 計算）

用法：python train_v2.py            比較並輸出最好的模型
     python train_v2.py --dry-run  只比較，不寫檔
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

DIR = Path(__file__).parent
ELO_K, ELO_INIT = 16, 1500  # K 以驗證集比較 8–48 後選定（12–16 最好）
ROLES = ['top', 'jungle', 'mid', 'bottom', 'support']
BARON_BUFF, ELDER_BUFF = 180, 150  # 秒

# 特徵定義需與 winprob.js 的 featuresV2() 一致（皆為 藍 − 紅）
FEATURES = ['minutes', 'gold', 'total_gold', 'kills', 'towers', 'inhibs', 'dragons', 'soul', 'elder', 'barons',
            'baron_active', 'elder_active', 'gold_trend', 'gold_top', 'gold_jungle', 'gold_mid', 'gold_bottom',
            'gold_support', 'levels', 'elo']
# 單調限制：這些優勢只會提高勝率（避免模型在資料稀少處學出反直覺的結果）
MONOTONE = {'comp_wr': 1, 'gold': 1, 'towers': 1, 'inhibs': 1, 'dragons': 1, 'soul': 1, 'elder': 1, 'barons': 1,
            'baron_active': 1, 'elder_active': 1, 'levels': 1, 'elo': 1}


# 陣容特徵（藍 − 紅）：5 隻英雄過去勝率（log-odds）的加總；每局只用「這局之前」的比賽統計，避免把答案算進特徵
# 也試過前期強度（15 分經濟差）、後期強度（長局勝率），太雜反而讓驗證分數變差，所以不用
DRAFT_FEATURES = ['comp_wr']
CHAMP_PRIOR = 60  # 貝氏平滑：樣本少的英雄往 50% 靠攏（20/60/150 驗證結果相近，取中間值）


class ChampStats:
    def __init__(self):
        self.s = defaultdict(lambda: [0, 0])  # 局數, 勝場

    def value(self, c):
        n, w = self.s[c]
        p = (w + CHAMP_PRIOR * 0.5) / (n + CHAMP_PRIOR)
        return math.log(p / (1 - p))

    def comp(self, blue, red):
        return [sum(self.value(c) for c in blue) - sum(self.value(c) for c in red)]

    def update(self, blue, red, blue_win):
        for champs, win in ((blue, blue_win), (red, not blue_win)):
            for c in champs:
                self.s[c][0] += 1
                self.s[c][1] += win

    def table(self):
        return {c: round(self.value(c), 4) for c, st in self.s.items() if st[0] >= 3}


def label_games(data):
    """推塔判定的勝方，與官方系列賽比分比對；回傳 {gameId: blueWin}（比分不符的系列賽排除）"""
    games, ok, bad, labels = data['games'], 0, 0, {}
    for m in data['matches'].values():
        gs = [games.get(g) for g in m['games']]
        if not gs or any(g is None for g in gs) or not m['wins']:
            continue
        won = defaultdict(int)
        for g in gs:
            won[g['teams'][0] if g['blueWin'] else g['teams'][1]] += 1
        if all(won.get(t, 0) == w for t, w in m['wins'].items()):
            ok += 1
            for gid in m['games']:
                labels[gid] = games[gid]['blueWin']
        else:
            bad += 1
    print(f'勝方標籤驗證：{ok} 場系列賽比分吻合、{bad} 場不符（排除）')
    return labels


def features(samples, k, elo_diff):
    """第 k 個樣本的特徵；k = -1 代表開局（只有 Elo）"""
    if k < 0:
        return [0] * (len(FEATURES) - 1) + [elo_diff / 100]
    s = samples[k]
    b, r = s['b'], s['r']
    gd = (b[0] - r[0]) / 1000
    prev = samples[max(0, k - 2)]
    trend = gd - (prev['b'][0] - prev['r'][0]) / 1000 if k >= 2 else 0
    reg = lambda t: [d for d in t[5] if d != 'elder']
    elders = lambda t: t[5].count('elder')

    def active(idx_fn, window):  # 最近 window 秒內有拿到 → 該隊 buff 生效中
        for j in range(k, -1, -1):
            if s['t'] - samples[j]['t'] > window:
                break
            pj = samples[j - 1] if j > 0 else None
            for sign, side in ((1, 'b'), (-1, 'r')):
                if pj and idx_fn(samples[j][side]) > idx_fn(pj[side]):
                    return sign
        return 0

    return [
        s['t'] / 60, gd, (b[0] + r[0]) / 1000, b[1] - r[1], b[2] - r[2], b[3] - r[3],
        len(reg(b)) - len(reg(r)), (len(reg(b)) >= 4) - (len(reg(r)) >= 4), elders(b) - elders(r), b[4] - r[4],
        active(lambda t: t[4], BARON_BUFF), active(elders, ELDER_BUFF), trend,
        *[(b[6][i][0] - r[6][i][0]) / 1000 for i in range(5)],
        sum(p[1] for p in b[6]) - sum(p[1] for p in r[6]),
        elo_diff / 100,
    ]


V1_FEATURES = ['gold', 'gold_x_t', 'kills', 'towers', 'towers_x_t', 'dragons', 'soul', 'elder', 'barons', 'inhibs',
               'elo', 'elo_x_early']


def v1_features(s, t, elo_diff):
    """v1 的特徵（用同一批樣本重訓 v1 方法，作為公平的比較基準）"""
    if s is None:
        x = {'bias': 1, 'elo': elo_diff / 100, 'elo_x_early': elo_diff / 100}
    else:
        b, r, tm = s['b'], s['r'], t / 600
        gd, td = (b[0] - r[0]) / 1000, b[2] - r[2]
        x = {'bias': 1, 'gold': gd, 'gold_x_t': gd * tm, 'kills': b[1] - r[1], 'towers': td, 'towers_x_t': td * tm,
             'dragons': len(b[5]) - len(r[5]), 'soul': (len(b[5]) >= 4) - (len(r[5]) >= 4),
             'elder': b[5].count('elder') - r[5].count('elder'), 'barons': b[4] - r[4], 'inhibs': b[3] - r[3],
             'elo': elo_diff / 100, 'elo_x_early': elo_diff / 100 * max(0, 1 - tm / 3)}
    return [x.get(f, 0) for f in V1_FEATURES]


def v1_prob(model, f):
    """目前網站上的 v1 模型（它的訓練資料包含驗證集，分數僅供參考）"""
    x = dict(zip(V1_FEATURES, f), bias=1)
    z = sum(w * x.get(name, 0) for name, w in zip(model['features'], model['weights']))
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


def report(name, y, p, t):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    line = f'{name:<18} log-loss {log_loss(y, p):.4f}  Brier {brier_score_loss(y, p):.4f}  準確率 {accuracy_score(y, p > .5):.1%}'
    for lo, hi in [(0, 1), (9, 11), (19, 21), (29, 31)]:
        m = (t >= lo) & (t < hi)
        if m.any():
            line += f'  | {lo}分 {accuracy_score(y[m], p[m] > .5):.0%}'
    print(line)
    return log_loss(y, p)


def export_trees(booster):
    """LightGBM → 精簡 JSON：每棵樹為 [特徵, 門檻, 左, 右] 陣列；子節點 < 0 代表葉子 ~idx"""
    out = []
    for tree in booster.dump_model()['tree_info']:
        feat, thr, left, right, leaves = [], [], [], [], []

        def walk(node):
            if 'leaf_value' in node:
                leaves.append(round(node['leaf_value'], 6))
                return ~(len(leaves) - 1)
            i = len(feat)
            # 門檻不可四捨五入：整數特徵的門檻常是 ±1e-35，四捨五入成 0 會讓剛好等於 0 的值走錯邊
            feat.append(node['split_feature']); thr.append(node['threshold']); left.append(0); right.append(0)
            left[i] = walk(node['left_child'])
            right[i] = walk(node['right_child'])
            return i
        root = walk(tree['tree_structure'])
        out.append({'f': feat, 't': thr, 'l': left, 'r': right, 'v': leaves, 'root': root})
    return out


def tree_predict(trees, x):
    z = 0.0
    for tr in trees:
        n = tr['root']
        while n >= 0:
            n = tr['l'][n] if x[tr['f'][n]] <= tr['t'][n] else tr['r'][n]
        z += tr['v'][~n]
    return 1 / (1 + math.exp(-z))


def main():
    data = json.loads((DIR / 'winprob_v2_samples.json').read_text(encoding='utf-8'))
    history = {g['id']: g for g in json.loads((DIR / 'history.json').read_text(encoding='utf-8'))['games']}
    labels = label_games(data)
    # 網站目前的 v1 模型（輸出 v2 後原檔會被覆蓋，所以優先讀備份）
    v1_path = DIR / 'winprob_model_v1.json'
    v1 = json.loads((v1_path if v1_path.exists() else DIR / 'winprob_model.json').read_text(encoding='utf-8'))

    # Elo（依時間順序；同一天同系列以 gameId 排序）
    order = sorted((gid for gid in labels if gid in history), key=lambda g: (history[g]['date'], g))
    elo, rows, champs = {}, [], ChampStats()
    for gid in order:
        g, bw = data['games'][gid], labels[gid]
        bt, rt = g['teams']
        eb, er = elo.get(bt, ELO_INIT), elo.get(rt, ELO_INIT)
        pb = [p['champ'] for p in history[gid]['blue']['picks']]
        pr = [p['champ'] for p in history[gid]['red']['picks']]
        rows.append((gid, eb - er, bw, champs.comp(pb, pr)))  # 先取「賽前」的英雄數據，再用本局結果更新
        champs.update(pb, pr, bw)
        exp = 1 / (1 + 10 ** ((er - eb) / 400))
        d = ELO_K * (bw - exp)
        elo[bt], elo[rt] = eb + d, er - d

    split = int(len(rows) * 0.85)
    X, X1, C, y, T, is_test = [], [], [], [], [], []
    for i, (gid, ed, bw, comp) in enumerate(rows):
        smp = data['games'][gid]['samples']
        for k in range(-1, len(smp)):
            X.append(features(smp, k, ed)); y.append(int(bw)); T.append(0 if k < 0 else smp[k]['t'] / 60)
            X1.append(v1_features(None if k < 0 else smp[k], 0 if k < 0 else smp[k]['t'], ed))
            C.append(comp)
            is_test.append(i >= split)
    X, X1, y, T, te = np.array(X), np.array(X1), np.array(y), np.array(T), np.array(is_test)
    P1 = np.array([v1_prob(v1, f) for f in X1])
    XD = np.hstack([X, np.array(C)])  # 加上陣容特徵
    tr = ~te
    print(f'{len(rows)} 局（訓練 {split}、驗證 {len(rows) - split}），樣本 {tr.sum()} / {te.sum()}，藍方勝率 {y.mean():.1%}')

    # 1. v1 方法用同一個訓練集重訓（公平基準）
    base = LogisticRegression(C=1.0, max_iter=2000).fit(X1[tr], y[tr])
    # 2. 邏輯斯迴歸（新特徵）
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], y[tr])
    # 3. LightGBM：先用訓練集內最後一段做 early stopping 決定樹數，再用整個訓練集重訓
    params = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                  monotone_constraints=[MONOTONE.get(f, 0) for f in FEATURES])
    idx = np.where(tr)[0]
    cut = idx[int(len(idx) * 0.85)]
    fit_m, val_m = tr & (np.arange(len(y)) < cut), tr & (np.arange(len(y)) >= cut)
    probe = lgb.train(params, lgb.Dataset(X[fit_m], y[fit_m]), 2000, valid_sets=[lgb.Dataset(X[val_m], y[val_m])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    n_trees = max(50, probe.best_iteration)
    gbm = lgb.train(params, lgb.Dataset(X[tr], y[tr]), n_trees)
    # 4. 加上陣容特徵
    params_d = dict(params, monotone_constraints=[MONOTONE.get(f, 0) for f in FEATURES + DRAFT_FEATURES])
    probe_d = lgb.train(params_d, lgb.Dataset(XD[fit_m], y[fit_m]), 2000, valid_sets=[lgb.Dataset(XD[val_m], y[val_m])],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    n_trees_d = max(50, probe_d.best_iteration)
    gbm_d = lgb.train(params_d, lgb.Dataset(XD[tr], y[tr]), n_trees_d)
    lr_d = LogisticRegression(C=1.0, max_iter=2000).fit(XD[tr], y[tr])

    print(f'\n驗證集（最近 {len(rows) - split} 局，LightGBM {n_trees} 棵樹）：')
    report('網站目前模型*', y[te], P1[te], T[te])
    ll1 = report('v1 方法重訓', y[te], base.predict_proba(X1[te])[:, 1], T[te])
    ll2 = report('邏輯斯迴歸 v2', y[te], lr.predict_proba(X[te])[:, 1], T[te])
    ll3 = report('LightGBM v2', y[te], gbm.predict(X[te]), T[te])
    ll4 = report('邏輯斯 v2+陣容', y[te], lr_d.predict_proba(XD[te])[:, 1], T[te])
    ll5 = report('LightGBM v2+陣容', y[te], gbm_d.predict(XD[te]), T[te])
    imp_d = dict(zip(FEATURES + DRAFT_FEATURES, gbm_d.feature_importance('gain')))
    print(f'陣容勝率重要度占比：{imp_d["comp_wr"] / sum(imp_d.values()):.1%}')
    imp = sorted(zip(FEATURES, gbm.feature_importance('gain')), key=lambda x: -x[1])
    print('LightGBM 特徵重要度：' + '、'.join(f'{f}' for f, _ in imp[:8]))
    print('* 網站目前模型的訓練資料包含這批驗證對局，分數偏高，僅供參考；公平比較看「v1 方法重訓」')

    if '--dry-run' in sys.argv:
        return
    best = min([('v1', ll1), ('lr', ll2), ('gbm', ll3), ('lr_d', ll4), ('gbm_d', ll5)], key=lambda x: x[1])[0]
    if best.endswith('_d'):  # 陣容版：特徵加上陣容，並輸出英雄數據表（用全部對局統計）
        best, X, lr, gbm, params, n_trees, ll2, ll3 = best[:-2], XD, lr_d, gbm_d, params_d, n_trees_d, ll4, ll5
        feats = FEATURES + DRAFT_FEATURES
    else:
        feats = FEATURES
    if best == 'v1':
        print('新特徵沒有比 v1 方法好，保留目前模型')
        return

    # 用全部資料重訓最終模型並輸出
    teams = {}
    for g in history.values():
        for side in ('blue', 'red'):
            teams[g[side]['teamId']] = g[side]['code']
    out = {'version': 2, 'features': feats, 'games': len(rows), 'elo': {k: round(v, 1) for k, v in elo.items()},
           'teams': teams, 'eloInit': ELO_INIT, 'buff': {'baron': BARON_BUFF, 'elder': ELDER_BUFF}}
    if feats != FEATURES:
        out['champs'] = champs.table()
    if best == 'gbm':
        final = lgb.train(params, lgb.Dataset(X, y), n_trees)
        trees = export_trees(final)
        chk = [abs(tree_predict(trees, x) - p) for x, p in zip(X[:2000], final.predict(X[:2000]))]
        print(f'樹結構匯出檢查：最大誤差 {max(chk):.2e}')
        out.update(type='gbm', trees=trees, validation={'logloss': round(ll3, 4), 'accuracy': round(float(accuracy_score(y[te], gbm.predict(X[te]) > .5)), 4)})
    else:
        final = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
        out.update(type='linear', weights=[round(float(w), 5) for w in final.coef_[0]], bias=round(float(final.intercept_[0]), 5),
                   validation={'logloss': round(ll2, 4), 'accuracy': round(float(accuracy_score(y[te], lr.predict_proba(X[te])[:, 1] > .5)), 4)})
    path = DIR / 'winprob_model.json'
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(f'輸出 {best} 模型 → {path}（{path.stat().st_size // 1024} KB）')


if __name__ == '__main__':
    main()
