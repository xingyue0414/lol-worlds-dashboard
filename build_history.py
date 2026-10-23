"""收集近期職業比賽的選角資料 → history.json，供面板的選角機率模型使用。

用法：python build_history.py [起始日期，預設 2026-01-01]
已抓過的對局會沿用 history.json 的快取，重跑只會補新對局。
（lolesports 的 livestats 只有選出的英雄，沒有禁用資料。）
"""
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GW = 'https://esports-api.lolesports.com/persisted/gw/'
FEED = 'https://feed.lolesports.com/livestats/v1/window/'
KEY = '0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z'
LEAGUES = ['worlds', 'msi', 'ewc_lol', 'lck', 'lpl', 'lec', 'lcs', 'lta_n', 'cblol-brazil', 'lcp', 'vcs', 'ljl-japan']
OUT = Path(__file__).with_name('history.json')
SINCE = sys.argv[1] if len(sys.argv) > 1 else '2026-01-01'


def get(url, headers=None):
    req = urllib.request.Request(url, headers={'User-Agent': 'worlds-dashboard', **(headers or {})})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read()
                return json.loads(body) if body else None
        except urllib.error.HTTPError as e:
            if e.code in (204, 404):
                return None
            if attempt == 2:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise


def gw(path):
    return get(GW + path, {'x-api-key': KEY})['data']


def league_matches(league_id):
    out, token = [], None
    while True:
        d = gw(f'getSchedule?hl=en-US&leagueId={league_id}' + (f'&pageToken={token}' if token else ''))['schedule']
        evs = d['events']
        out += [e for e in evs if e['state'] == 'completed' and e.get('match') and e['startTime'] >= SINCE]
        token = d['pages']['older']
        if not token or not evs or min(e['startTime'] for e in evs) < SINCE:
            return out


def game_picks(game_id):
    w = get(FEED + game_id)
    if not w or 'gameMetadata' not in w:
        return None
    m = w['gameMetadata']
    sides = {}
    for side in ('blue', 'red'):
        tm = m[side + 'TeamMetadata']
        sides[side] = {
            'teamId': tm['esportsTeamId'],
            'picks': [{'playerId': p.get('esportsPlayerId'), 'name': p['summonerName'].split(' ', 1)[-1],
                       'champ': p['championId'], 'role': p['role']} for p in tm['participantMetadata']],
        }
    return {'patch': m.get('patchVersion'), **sides}


def main():
    cache = {}
    if OUT.exists():
        cache = {g['id']: g for g in json.loads(OUT.read_text(encoding='utf-8'))['games']}
    leagues = {l['slug']: l for l in gw('getLeagues?hl=en-US')['leagues']}

    with ThreadPoolExecutor(12) as pool:
        matches = []
        for slug, ms in zip(LEAGUES, pool.map(lambda s: league_matches(leagues[s]['id']) if s in leagues else [], LEAGUES)):
            print(f'{slug}: {len(ms)} 場比賽', flush=True)
            matches += [(slug, e) for e in ms]

        def details(item):
            slug, e = item
            try:
                ev = gw(f'getEventDetails?hl=en-US&id={e["match"]["id"]}')['event']
            except Exception:
                return []
            teams = {t['id']: t['code'] for t in ev['match']['teams']}
            return [{'id': g['id'], 'league': slug, 'date': e['startTime'], 'teams': teams}
                    for g in ev['match']['games'] if g['state'] == 'completed']

        todo = [g for gs in pool.map(details, matches) for g in gs]
        new = [g for g in todo if g['id'] not in cache]
        print(f'對局 {len(todo)}，需新抓 {len(new)}', flush=True)

        def fetch(g):
            try:
                p = game_picks(g['id'])
            except Exception:
                return None
            if not p:
                return None
            for side in ('blue', 'red'):
                p[side]['code'] = g['teams'].get(p[side]['teamId'], '?')
            return {'id': g['id'], 'league': g['league'], 'date': g['date'], **p}

        for i, r in enumerate(pool.map(fetch, new), 1):
            if r:
                cache[r['id']] = r
            if i % 50 == 0:
                print(f'  {i}/{len(new)}', flush=True)

    games = sorted(cache.values(), key=lambda g: g['date'])
    OUT.write_text(json.dumps({'since': SINCE, 'games': games}, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(f'完成：{len(games)} 局 → {OUT}')


if __name__ == '__main__':
    main()
