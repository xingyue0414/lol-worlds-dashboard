"""勝率模型 v2 的訓練資料：每分鐘抽一個場上狀態（扣除暫停時間），並用官方系列賽比分驗證勝方。

輸出 winprob_v2_samples.json（快取，重跑只補新對局）。之後用 train_v2.py 訓練。
用法：python collect_v2.py
"""
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

from build_history import LEAGUES, get, gw, league_matches

FEED = 'https://feed.lolesports.com/livestats/v1/window/'
DIR = Path(__file__).parent
OUT = DIR / 'winprob_v2_samples.json'
ROLES = ['top', 'jungle', 'mid', 'bottom', 'support']
STEP = 60  # 秒


def ts(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00'))


def iso(dt):
    dt = dt.replace(microsecond=0, second=dt.second // 10 * 10)
    return dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')


def last_frame(game_id, at):
    w = get(FEED + game_id + f'?startingTime={iso(at)}')
    return w['frames'][-1] if w and w.get('frames') else None


def team_state(f, side, role_of):
    t = f[side + 'Team']
    players = {role_of.get(p['participantId']): p for p in t['participants']}
    return [t['totalGold'], t['totalKills'], t['towers'], t['inhibitors'], t['barons'], t['dragons'],
            [[players[r]['totalGold'], players[r]['level'], players[r]['creepScore']] if r in players else [0, 1, 0] for r in ROLES]]


def collect(game_id):
    first = get(FEED + game_id)
    if not first or not first.get('frames') or 'gameMetadata' not in first:
        return None
    m = first['gameMetadata']
    role_of = {p['participantId']: p['role'] for side in ('blue', 'red') for p in m[side + 'TeamMetadata']['participantMetadata']}
    t0 = ts(first['frames'][0]['rfc460Timestamp'])
    prev_ts, prev_state, gt = t0, first['frames'][-1]['gameState'], 0.0
    samples, last_ok, k, final = [], t0, 1, None
    while k < 120:
        at = t0 + timedelta(seconds=k * STEP)
        f = last_frame(game_id, at)
        if not f:  # 已超過結束時間：二分搜尋最後一個 frame
            lo, hi = last_ok, at
            while (hi - lo).total_seconds() > 10:
                mid = lo + (hi - lo) / 2
                g = last_frame(game_id, mid)
                if g:
                    lo, final = mid, g
                else:
                    hi = mid
            final = final or last_frame(game_id, lo)
            break
        last_ok, k = at, k + 1
        if f['gameState'] == 'finished':
            final = f
            break
        cur = ts(f['rfc460Timestamp'])
        if cur <= prev_ts:  # 暫停中：feed 停在暫停前最後一個 frame
            prev_state = 'paused'
            continue
        delta = (cur - prev_ts).total_seconds()
        if prev_state == 'paused' or delta > 90:  # 剛恢復：真正進行的時間不會超過一個取樣間隔
            delta = min(delta, STEP)
        gt += delta
        prev_ts, prev_state = cur, f['gameState']
        samples.append({'t': round(gt), 'b': team_state(f, 'blue', role_of), 'r': team_state(f, 'red', role_of)})
    if not final:
        return None
    fb, fr = final['blueTeam'], final['redTeam']
    key = lambda t: (t['towers'], t['inhibitors'], t['totalGold'])
    return {'samples': samples, 'blueWin': key(fb) > key(fr),
            'teams': [m['blueTeamMetadata']['esportsTeamId'], m['redTeamMetadata']['esportsTeamId']]}


def safe(fn, x):
    try:
        return fn(x)
    except Exception:
        return None


def main():
    history = {g['id']: g for g in json.loads((DIR / 'history.json').read_text(encoding='utf-8'))['games']}
    cache = json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else {'games': {}, 'matches': {}}
    leagues = {l['slug']: l for l in gw('getLeagues?hl=en-US')['leagues']}

    with ThreadPoolExecutor(16) as pool:
        # 系列賽結果（各隊贏幾局），用來驗證每局的勝方判定
        events = [e for ms in pool.map(lambda s: safe(league_matches, leagues[s]['id']) or [] if s in leagues else [], LEAGUES) for e in ms]
        todo = [e['match']['id'] for e in events if e['match']['id'] not in cache['matches']]

        def details(mid):
            m = gw(f'getEventDetails?hl=en-US&id={mid}')['event']['match']
            return mid, {'games': [g['id'] for g in m['games'] if g['state'] == 'completed'],
                         'wins': {t['id']: t['result']['gameWins'] for t in m['teams'] if t.get('result')}}
        for r in pool.map(lambda x: safe(details, x), todo):
            if r:
                cache['matches'][r[0]] = r[1]
        print(f'系列賽 {len(cache["matches"])} 場', flush=True)

        new = [gid for gid in history if gid not in cache['games']]
        print(f'需抓取 {len(new)} 局', flush=True)
        for i, (gid, r) in enumerate(zip(new, pool.map(lambda x: safe(collect, x), new)), 1):
            cache['games'][gid] = r
            if i % 100 == 0:
                print(f'  {i}/{len(new)}', flush=True)
                OUT.write_text(json.dumps(cache, separators=(',', ':')), encoding='utf-8')
    OUT.write_text(json.dumps(cache, separators=(',', ':')), encoding='utf-8')
    print(f'完成：{sum(1 for v in cache["games"].values() if v)} 局有資料 → {OUT}')


if __name__ == '__main__':
    main()
