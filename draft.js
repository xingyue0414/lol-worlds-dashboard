// 選角（BP）面板 + 即時選用機率模型
// 資料來源：history.json（build_history.py 產生，近期職業賽每局 10 位選手的英雄/位置）
// lolesports 即時 feed 在選角階段沒有資料，所以 BP 由使用者看直播點選輸入；開局後會用 feed 自動補齊選角。
const Draft = (() => {
  const ROLES = ['top', 'jungle', 'mid', 'bottom', 'support'];
  const BASE_ORDER = [ // 職業賽標準 BP 順序（blue = 先選方）
    ['blue', 'ban'], ['red', 'ban'], ['blue', 'ban'], ['red', 'ban'], ['blue', 'ban'], ['red', 'ban'],
    ['blue', 'pick'], ['red', 'pick'], ['red', 'pick'], ['blue', 'pick'], ['blue', 'pick'], ['red', 'pick'],
    ['red', 'ban'], ['blue', 'ban'], ['red', 'ban'], ['blue', 'ban'],
    ['red', 'pick'], ['blue', 'pick'], ['blue', 'pick'], ['red', 'pick'],
  ];
  // 先選權可能不在藍方（例如由隊伍自選先選或選邊），此時把順序裡的藍紅對調
  let ORDER = BASE_ORDER;
  const setFirst = side => { ORDER = side === 'red' ? BASE_ORDER.map(([s, k]) => [s === 'blue' ? 'red' : 'blue', k]) : BASE_ORDER; };
  const HALF_LIFE_DAYS = 25; // 越近的比賽權重越高
  const PLAYER_HALF_LIFE_DAYS = 120; // 選手英雄池變化比版本慢，用較長的半衰期（讓久未出賽的替補也有資料）
  const K_TEAM = 40, K_PLAYER = 30; // 貝氏平滑強度：樣本少時往上一層（全體 → 隊伍 → 選手）靠攏；數值以回測近 200 局調出

  let H = null;          // 統計結果
  let st = null;         // 目前對局的 BP 狀態 {gameId, teams, actions:[], locked:Set, fearless}
  let filterText = '';

  // ---------- 歷史統計 ----------
  async function loadHistory() {
    if (H) return H;
    try {
      const data = await (await fetch('history.json', { cache: 'no-cache' })).json();
      H = buildStats(data.games);
    } catch {
      H = buildStats([]);
      H.missing = true;
    }
    return H;
  }

  function buildStats(games) {
    const now = Date.now();
    const meta = Object.fromEntries(ROLES.map(r => [r, { tot: 0, c: {} }]));
    const champRole = {}, team = {}, player = {}, roster = {};
    const add = (o, k, w) => { o[k] = (o[k] || 0) + w; };
    for (const g of games) {
      const days = (now - Date.parse(g.date)) / 864e5;
      const boost = g.league === 'worlds' ? 1.5 : 1;
      const w = Math.pow(0.5, Math.max(0, days) / HALF_LIFE_DAYS) * boost;
      const wp = Math.pow(0.5, Math.max(0, days) / PLAYER_HALF_LIFE_DAYS) * boost;
      for (const side of ['blue', 'red']) {
        const t = g[side];
        team[t.teamId] ??= Object.fromEntries(ROLES.map(r => [r, { tot: 0, c: {} }]));
        roster[t.teamId] ??= { date: '', code: t.code, players: {} };
        const ro = roster[t.teamId];
        for (const p of t.picks) {
          if (!ROLES.includes(p.role)) continue;
          meta[p.role].tot += w; add(meta[p.role].c, p.champ, w);
          team[t.teamId][p.role].tot += w; add(team[t.teamId][p.role].c, p.champ, w);
          champRole[p.champ] ??= {}; add(champRole[p.champ], p.role, w);
          if (p.playerId) {
            player[p.playerId] ??= { name: p.name, tot: 0, n: {}, c: {} };
            const pl = player[p.playerId];
            pl.tot += wp; add(pl.c, p.champ, wp); add(pl.n, p.champ, 1); pl.name = p.name;
          }
          if (g.date >= ro.date) ro.players[p.role] = { id: p.playerId, name: p.name };
        }
        if (g.date >= ro.date) { ro.date = g.date; ro.code = t.code; }
      }
    }
    return { games: games.length, meta, champRole, team, player, roster };
  }

  // ---------- 模型 ----------
  const allChamps = () => Object.keys(dd.champs);
  function roleDist(c) {
    const r = H.champRole[c];
    const tot = r ? ROLES.reduce((s, x) => s + (r[x] || 0), 0) : 0;
    return Object.fromEntries(ROLES.map(x => [x, tot ? ((r[x] || 0) + 0.05) / (tot + 0.25) : 0.2]));
  }

  // 已選英雄依序分配位置（貪婪：挑該英雄最常打、且尚未被占的位置）
  function assignRoles(picks) {
    const open = new Set(ROLES), out = [];
    for (const c of picks) {
      const rd = roleDist(c);
      const role = [...open].sort((a, b) => rd[b] - rd[a])[0];
      if (role) open.delete(role);
      out.push({ champ: c, role });
    }
    return { assigned: out, open: [...open] };
  }

  function unavailable() {
    const s = new Set(st.actions.filter(Boolean));
    if (st.fearless) for (const c of st.locked) s.add(c);
    return s;
  }

  // 某隊下一手選用各英雄的機率分布
  function pickDist(side, banned) {
    const teamId = st.teams[side].id;
    const picks = st.actions.filter((c, i) => c && ORDER[i][0] === side && ORDER[i][1] === 'pick');
    const { open } = assignRoles(picks);
    const roles = open.length ? open : ROLES;
    const T = H.team[teamId];
    const champs = allChamps().filter(c => !banned.has(c));
    const nMeta = champs.length;
    const score = {}, why = {};
    for (const c of champs) {
      let s = 0, best = 0;
      for (const role of roles) {
        const m = H.meta[role];
        const pMeta = ((m.c[c] || 0) + 0.02) / (m.tot + 0.02 * nMeta);
        const t = T?.[role];
        const pTeam = t ? ((t.c[c] || 0) + K_TEAM * pMeta) / (t.tot + K_TEAM) : pMeta;
        const who = st.roster[side][role], pl = H.player[who?.id];
        // 沒有歷史資料的選手（例如替補上場）：隊伍在該位置的歷史其實是前一位選手的英雄池，所以和版本熱門度各半
        const pPlayer = pl ? ((pl.c[c] || 0) + K_PLAYER * pTeam) / (pl.tot + K_PLAYER) : who?.id ? (pTeam + pMeta) / 2 : pTeam;
        const v = pPlayer / roles.length;
        s += v;
        if (v > best) {
          best = v;
          why[c] = pl && pl.n[c] ? `${who.name} 今年用過 ${pl.n[c]} 次` : (m.c[c] ? `${roleZh(role)}版本強勢` : '');
        }
      }
      score[c] = s;
    }
    return { dist: normalize(score), why, open };
  }

  // 某隊下一手禁用：以「對手下一手可能選用」為主、混合版本熱門度
  function banDist(side, banned) {
    const opp = side === 'blue' ? 'red' : 'blue';
    const { dist, why } = pickDist(opp, banned);
    const score = {};
    for (const c of Object.keys(dist)) {
      const m = ROLES.reduce((s, r) => s + (H.meta[r].c[c] || 0) / (H.meta[r].tot || 1), 0);
      score[c] = 0.7 * dist[c] + 0.3 * m / ROLES.length;
    }
    return { dist: normalize(score), why };
  }

  function normalize(o) {
    const sum = Object.values(o).reduce((a, b) => a + b, 0) || 1;
    return Object.fromEntries(Object.entries(o).map(([k, v]) => [k, v / sum]));
  }

  // 本局出場機率：剩餘選用次數下至少被一隊選中的機率（扣掉剩餘禁用可能被 ban 掉的機率）
  function inGameProb(banned) {
    const step = st.actions.length;
    const rest = ORDER.slice(step);
    const left = side => rest.filter(([s, a]) => s === side && a === 'pick').length;
    const bansLeft = rest.filter(([, a]) => a === 'ban').length;
    const pb = pickDist('blue', banned).dist, pr = pickDist('red', banned).dist;
    const bans = bansLeft ? banDist(rest.find(([, a]) => a === 'ban')[0], banned).dist : {};
    const out = {};
    for (const c of Object.keys(pb)) {
      const notPicked = Math.pow(1 - pb[c], left('blue')) * Math.pow(1 - (pr[c] || 0), left('red'));
      const survive = Math.pow(1 - (bans[c] || 0), bansLeft);
      out[c] = (1 - notPicked) * survive;
    }
    return out;
  }

  // ---------- 狀態 ----------
  const key = id => `draft:${id}`;
  function save() { try { localStorage.setItem(key(st.gameId), JSON.stringify({ actions: st.actions, fearless: st.fearless, first: st.first })); } catch {} }

  async function open(g, allGames) {
    await loadHistory();
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(key(g.gameId))); } catch {}
    st = { gameId: g.gameId, teams: g.teams, actions: saved?.actions || [], fearless: saved?.fearless ?? true, first: saved?.first || 'blue', locked: new Set(), fromFeed: false,
      roster: { blue: defaultRoster(g.teams.blue), red: defaultRoster(g.teams.red) }, official: { blue: [], red: [] } };
    setFirst(st.first);
    $('#draftCard').hidden = false;
    render();
    loadOfficial(st);
    loadLocked(st, g, allGames); // 不等它：open() 結束時 st 就已就緒，呼叫端才能接著套用 feed
  }

  // Fearless：同系列賽前面幾局已選過的英雄
  async function loadLocked(my, g, allGames) {
    const prev = allGames.filter(x => x.matchId === g.matchId && x.gameId !== g.gameId && x.gameNumber < g.gameNumber);
    const metas = await Promise.all(prev.map(x => feed('window', x.gameId).catch(() => null)));
    if (my !== st) return;
    for (const w of metas) for (const tm of [w?.gameMetadata?.blueTeamMetadata, w?.gameMetadata?.redTeamMetadata])
      for (const p of tm?.participantMetadata || []) st.locked.add(p.championId);
    render();
  }

  // ---------- 先發名單 ----------
  // 預設：該隊最近一局的先發；使用者手動改過的位置（存在 localStorage）優先
  const rosterKey = teamId => `roster:${teamId}`;
  function defaultRoster(team) {
    const out = {}, ro = H.roster[team.id];
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem(rosterKey(team.id))) || {}; } catch {}
    for (const r of ROLES) {
      const p = saved[r] || ro?.players[r];
      out[r] = p ? { id: p.id, name: cleanName(p.name, team.code), manual: !!saved[r] } : null;
    }
    return out;
  }
  function setPlayer(side, role, player) {
    const team = st.teams[side];
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem(rosterKey(team.id))) || {}; } catch {}
    if (player) saved[role] = { id: player.id, name: player.name }; else delete saved[role];
    try { localStorage.setItem(rosterKey(team.id), JSON.stringify(saved)); } catch {}
    st.roster[side] = defaultRoster(team);
    render();
  }
  // 官方登錄名單（含替補），作為下拉選項
  async function loadOfficial(my) {
    for (const side of ['blue', 'red']) {
      try {
        const t = (await gw('getTeams', { id: my.teams[side].id })).teams[0];
        my.official[side] = (t?.players || []).map(p => ({ id: p.id, name: p.summonerName, role: p.role }));
      } catch {}
    }
    if (my === st) render();
  }

  // 開局後 feed 給出雙方英雄 → 補齊尚未輸入的選角
  function fromFeed(meta, gameId) {
    if (!st || st.fromFeed || st.gameId !== gameId) return;
    st.fromFeed = true;
    // 實際上場的先發覆蓋預測名單
    for (const [side, tm] of [['blue', meta.blueTeamMetadata], ['red', meta.redTeamMetadata]])
      for (const p of tm.participantMetadata)
        if (ROLES.includes(p.role)) st.roster[side][p.role] = { id: p.esportsPlayerId, name: cleanName(p.summonerName, st.teams[side].code), fromFeed: true };
    $('#draftCard').open = false; // 開局後收合，把畫面留給比賽資料
    const want = { blue: meta.blueTeamMetadata.participantMetadata.map(p => p.championId), red: meta.redTeamMetadata.participantMetadata.map(p => p.championId) };
    const entered = new Set(st.actions.filter(Boolean));
    const queue = { blue: want.blue.filter(c => !entered.has(c)), red: want.red.filter(c => !entered.has(c)) };
    const actions = [...st.actions];
    for (let i = actions.length; i < ORDER.length; i++) {
      const [side, kind] = ORDER[i];
      actions.push(kind === 'pick' ? (queue[side].shift() || null) : null);
    }
    st.actions = actions;
    save(); render();
  }

  function act(champ) {
    if (!st || st.actions.length >= ORDER.length) return;
    if (champ && unavailable().has(champ)) return;
    st.actions.push(champ);
    filterText = ''; $('#draftSearch').value = '';
    save(); render();
  }

  // ---------- BP 模擬 ----------
  // 以目前 BP 為起點，用模型逐手推演剩下的禁選。pickDist/banDist 讀的是 st.actions，模擬時暫時替換
  function withActions(actions, fn) {
    const saved = st.actions;
    st.actions = actions;
    try { return fn(); } finally { st.actions = saved; }
  }
  function nextDist(actions) {
    return withActions(actions, () => {
      const [side, kind] = ORDER[actions.length], banned = unavailable();
      return (kind === 'pick' ? pickDist(side, banned) : banDist(side, banned)).dist;
    });
  }
  function sample(dist) {
    let r = Math.random();
    for (const [c, p] of Object.entries(dist)) if ((r -= p) <= 0) return c;
    return Object.keys(dist)[0];
  }
  // 逐手取最高機率 → 最可能的一套 BP；另外隨機抽樣 N 次統計各英雄被選/被禁的頻率
  async function simulate(N = 300) {
    const start = [...st.actions];
    let greedy = [...start];
    while (greedy.length < ORDER.length) greedy.push(topN(nextDist(greedy), 1)[0][0]);
    const freq = { blue: { pick: {}, ban: {} }, red: { pick: {}, ban: {} } };
    for (let n = 0; n < N; n++) {
      const a = [...start];
      while (a.length < ORDER.length) {
        const c = sample(nextDist(a));
        const [side, kind] = ORDER[a.length];
        freq[side][kind][c] = (freq[side][kind][c] || 0) + 1 / N;
        a.push(c);
      }
      if (n % 25 === 24) { $('#draftSim').innerHTML = `<div class="muted">模擬中… ${n + 1}/${N}</div>`; await new Promise(r => setTimeout(r)); }
    }
    return { greedy, freq, N, from: start.length };
  }

  function renderSim(res) {
    const T = st.teams;
    const side = s => {
      const idx = kind => ORDER.map((o, i) => [o, i]).filter(([[x, k]]) => x === s && k === kind).map(([, i]) => i);
      const cell = (i, kind) => {
        const c = res.greedy[i], given = i < res.from;
        return `<div class="slot ${kind} ${given ? '' : 'sim'}">${c ? icon(c) : '<span class="muted">—</span>'}${kind === 'pick' && c ? `<span>${esc(champName(c))}</span>` : ''}</div>`;
      };
      const tops = kind => topN(res.freq[s][kind], kind === 'pick' ? 10 : 8)
        .map(([c, p]) => `<span class="simchip" title="${esc(champName(c))}">${icon(c)}${pct(p)}</span>`).join('');
      return `<div class="dside ${s}"><div class="dteam">${esc(T[s].code)}</div>
        <div class="bans">${idx('ban').map(i => cell(i, 'ban')).join('')}</div><div class="picks">${idx('pick').map(i => cell(i, 'pick')).join('')}</div>
        ${res.from < ORDER.length ? `<div class="simfreq"><div class="muted">被選頻率</div>${tops('pick')}</div><div class="simfreq"><div class="muted">被禁頻率</div>${tops('ban')}</div>` : ''}</div>`;
    };
    $('#draftSim').innerHTML = `<h3>🎲 模擬結果：最可能的 BP（虛線框 = 模擬）</h3><div class="simsides">${side('blue')}${side('red')}</div>
      <div class="muted" style="font-size:12px">上方是每一手都取最高機率的單一推演；「頻率」是隨機模擬 ${res.N} 次後，各英雄出現在該隊選/禁的比例。
      模型只看選手與隊伍的英雄池、版本熱門度與缺的位置，不考慮克制與陣容搭配。</div>`;
  }

  // ---------- 畫面 ----------
  const roleZh = r => ROLE[r] || r;
  const pct = p => p >= 0.995 ? '>99%' : p < 0.001 ? '<0.1%' : (p * 100).toFixed(p < 0.1 ? 1 : 0) + '%';
  const icon = c => `<img src="${champImg(c)}" alt="" title="${esc(champName(c))}">`;

  function render() {
    if (!st || !H) return;
    const step = st.actions.length, done = step >= ORDER.length;
    const banned = unavailable();
    const T = st.teams;

    // BP 欄位
    const col = side => {
      const slots = kind => ORDER.map((o, i) => [o, i]).filter(([[s, k]]) => s === side && k === kind).map(([, i]) => {
        const c = st.actions[i], cur = i === step;
        return `<div class="slot ${kind} ${cur ? 'cur' : ''}">${c ? icon(c) : (i < step ? '<span class="muted">—</span>' : '')}${kind === 'pick' && c ? `<span>${esc(champName(c))}</span>` : ''}</div>`;
      }).join('');
      // 每個位置一個下拉選單：官方名單中該位置的選手 + 目前預測的先發
      const pick = r => {
        const cur = st.roster[side][r];
        const opts = new Map();
        if (cur) opts.set(cur.id, cur.name);
        for (const p of st.official[side]) if (p.role === r || p.id === cur?.id) opts.set(p.id, p.name);
        for (const p of st.official[side]) if (!opts.has(p.id)) opts.set(p.id, `${p.name}（${roleZh(p.role)}）`);
        const noData = cur?.id && !H.player[cur.id];
        return `<select class="rsel ${cur?.manual ? 'manual' : ''}" data-side="${side}" data-role="${r}" title="${roleZh(r)}${noData ? '：近期沒有出賽資料' : ''}">` +
          [...opts].map(([id, n]) => `<option value="${esc(id)}" ${id === cur?.id ? 'selected' : ''}>${esc(n)}${id === cur?.id && noData ? ' *' : ''}</option>`).join('') + '</select>';
      };
      return `<div class="dside ${side}"><div class="dteam">${esc(T[side].code)} <span class="rsels">${ROLES.map(pick).join('')}</span></div>
        <div class="bans">${slots('ban')}</div><div class="picks">${slots('pick')}</div></div>`;
    };
    $('#draftSides').innerHTML = col('blue') + col('red');
    $('#pregame').innerHTML = WP.model ? wpBar(WP.pregame(T), T, `賽前勝率（Elo ${WP.elo(T.blue)} : ${WP.elo(T.red)}）`) : '';

    // 下一手預測
    let head = '', list = '';
    if (!done) {
      const [side, kind] = ORDER[step];
      const nth = ORDER.slice(0, step + 1).filter(([s, k]) => s === side && k === kind).length;
      const { dist, why, open } = kind === 'pick' ? pickDist(side, banned) : banDist(side, banned);
      head = `下一手：<b class="${side}">${esc(T[side].code)}</b> 第 ${nth} ${kind === 'pick' ? '選' : '禁'}` +
        (kind === 'pick' && open ? `<span class="muted">（缺：${open.map(roleZh).join('、')}）</span>` : '');
      list = topN(dist, 10).map(([c, p]) => bar(c, p, why[c], side)).join('');
    } else {
      head = 'BP 完成';
    }
    $('#draftNext').innerHTML = `<h3>${head}</h3>${list}`;
    const ig = done ? {} : inGameProb(banned);
    $('#draftInGame').innerHTML = done ? '' : `<h3>本局出場機率</h3>${topN(ig, 10).map(([c, p]) => bar(c, p, roleZh(topRole(c)))).join('')}`;

    // 英雄點選格
    const f = filterText.trim().toLowerCase();
    const order = done ? allChamps() : allChamps().sort((a, b) => (ig[b] || 0) - (ig[a] || 0));
    $('#champGrid').innerHTML = order.filter(c => !f || c.toLowerCase().includes(f) || champName(c).includes(f) || dd.champs[c]?.title?.includes(f))
      .map(c => `<button class="cg ${banned.has(c) ? 'off' : ''}" data-c="${c}" title="${esc(champName(c))}${st.locked.has(c) ? '（Fearless 已用過）' : ''}">${icon(c)}</button>`).join('');
    $('#fearless').checked = st.fearless;
    $('#firstPick').innerHTML = ['blue', 'red'].map(s => `<option value="${s}" ${st.first === s ? 'selected' : ''}>先選：${esc(T[s].code)}（${s === 'blue' ? '藍' : '紅'}方）</option>`).join('');
    $('#draftInfo').textContent = H.missing ? '⚠️ 找不到 history.json，請先執行 python build_history.py' :
      `模型資料：${H.games} 局職業賽` + (st.locked.size ? ` · Fearless 鎖定 ${st.locked.size} 隻` : '');
  }

  const topRole = c => { const r = roleDist(c); return ROLES.reduce((a, b) => r[a] >= r[b] ? a : b); };
  const topN = (o, n) => Object.entries(o).sort((a, b) => b[1] - a[1]).slice(0, n);
  const bar = (c, p, note, side) => `<div class="pbar">${icon(c)}<div class="pb-main"><div class="pb-top"><span>${esc(champName(c))}</span><b>${pct(p)}</b></div>
    <div class="pb-track"><i class="${side || ''}" style="width:${Math.min(100, p * 200)}%"></i></div>${note ? `<div class="muted pb-note">${esc(note)}</div>` : ''}</div></div>`;

  function init() {
    $('#champGrid').onclick = e => { const b = e.target.closest('.cg'); if (b) act(b.dataset.c); };
    $('#draftSearch').oninput = e => { filterText = e.target.value; render(); };
    $('#draftSearch').onkeydown = e => { if (e.key === 'Enter') { const b = $('#champGrid .cg:not(.off)'); if (b) act(b.dataset.c); } };
    $('#draftUndo').onclick = () => { if (st?.actions.length) { st.actions.pop(); st.fromFeed = false; save(); render(); } };
    $('#draftSkip').onclick = () => act(null);
    $('#draftReset').onclick = () => { if (st && confirm('清空這局的 BP？')) { st.actions = []; st.fromFeed = false; save(); render(); } };
    $('#fearless').onchange = e => { if (st) { st.fearless = e.target.checked; save(); render(); } };
    $('#firstPick').onchange = e => {
      if (!st) return;
      st.first = e.target.value; setFirst(st.first);
      save(); render(); $('#draftSim').innerHTML = '';
    };
    $('#draftSides').onchange = e => {
      const sel = e.target.closest('.rsel'); if (!sel || !st) return;
      const { side, role } = sel.dataset;
      const p = st.official[side].find(x => x.id === sel.value);
      setPlayer(side, role, p ? { id: p.id, name: p.name } : null);
    };
    $('#draftSimBtn').onclick = async () => {
      if (!st || !H || H.missing) return;
      const btn = $('#draftSimBtn'); btn.disabled = true;
      try { renderSim(await simulate()); } finally { btn.disabled = false; }
    };
  }

  return { init, open, fromFeed, render };
})();
