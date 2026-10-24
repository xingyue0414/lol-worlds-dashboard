// 即時勝率：讀取訓練好的模型（winprob_model.json）
//   v1：build_winprob.py 的邏輯斯迴歸（只看當下場面）
//   v2：train_v2.py 的 LightGBM 或邏輯斯迴歸（另含 buff 生效、經濟趨勢、各路經濟），特徵需與 train_v2.py 的 features() 一致
const WP = (() => {
  let model = null;
  const ROLES = ['top', 'jungle', 'mid', 'bottom', 'support'];

  async function load() {
    if (model) return; // 尚未載入成功就每次重試（模型可能還在訓練中）
    try { model = await (await fetch('winprob_model.json', { cache: 'no-cache' })).json(); }
    catch { model = false; }
    const info = document.querySelector('#wpInfo');
    if (info) info.textContent = model
      ? `（${model.games} 局訓練 · 驗證準確率 ${(model.validation.accuracy * 100).toFixed(0)}%）`
      : '（找不到 winprob_model.json，請先執行 python build_winprob.py）';
  }

  const eloOf = team => model.elo[team?.id] ?? model.eloInit;
  const sigmoid = z => 1 / (1 + Math.exp(-z));

  // ---------- v1 ----------
  function featuresV1(f, tSec, eloDiff) {
    const b = f.blueTeam, r = f.redTeam, tm = tSec / 600;
    const gd = (b.totalGold - r.totalGold) / 1000, td = b.towers - r.towers;
    const elder = side => side.dragons.filter(d => d === 'elder').length;
    return {
      bias: 1, gold: gd, gold_x_t: gd * tm, kills: b.totalKills - r.totalKills, towers: td, towers_x_t: td * tm,
      dragons: b.dragons.length - r.dragons.length, soul: (b.dragons.length >= 4) - (r.dragons.length >= 4),
      elder: elder(b) - elder(r), barons: b.barons - r.barons, inhibs: b.inhibitors - r.inhibitors,
      elo: eloDiff / 100, elo_x_early: eloDiff / 100 * Math.max(0, 1 - tm / 3),
    };
  }
  const probV1 = (frame, tSec, teams) => {
    const x = featuresV1(frame, tSec, eloOf(teams.blue) - eloOf(teams.red));
    return sigmoid(model.features.reduce((s, k, i) => s + model.weights[i] * x[k], 0));
  };

  // ---------- v2 ----------
  const regular = t => t.dragons.filter(d => d !== 'elder').length;
  const elders = t => t.dragons.filter(d => d === 'elder').length;
  const gdOf = f => (f.blueTeam.totalGold - f.redTeam.totalGold) / 1000;

  // snaps：{gt(ms), w(window frame)} 依時間排序；i：要計算的那一筆；roleOf：participantId → 位置
  function featuresV2(snaps, i, roleOf, eloDiff) {
    const s = snaps[i], f = s.w, b = f.blueTeam, r = f.redTeam;
    let prev = null; // 2 分鐘前的狀態（經濟趨勢）
    for (let j = i; j >= 0; j--) if (s.gt - snaps[j].gt >= 120000) { prev = snaps[j]; break; }
    const active = (count, window) => { // 最近 window 秒內拿到 → 該隊 buff 生效中
      for (let j = i; j > 0 && s.gt - snaps[j].gt <= window * 1000; j--) {
        const a = snaps[j - 1].w, c = snaps[j].w;
        if (count(c.blueTeam) > count(a.blueTeam)) return 1;
        if (count(c.redTeam) > count(a.redTeam)) return -1;
      }
      return 0;
    };
    const byRole = t => Object.fromEntries(t.participants.map(p => [roleOf[p.participantId], p]));
    const pb = byRole(b), pr = byRole(r);
    const lv = t => t.participants.reduce((a, p) => a + p.level, 0);
    return {
      minutes: s.gt / 60000, gold: gdOf(f), total_gold: (b.totalGold + r.totalGold) / 1000,
      kills: b.totalKills - r.totalKills, towers: b.towers - r.towers, inhibs: b.inhibitors - r.inhibitors,
      dragons: regular(b) - regular(r), soul: (regular(b) >= 4) - (regular(r) >= 4), elder: elders(b) - elders(r),
      barons: b.barons - r.barons,
      baron_active: active(t => t.barons, model.buff.baron), elder_active: active(elders, model.buff.elder),
      gold_trend: prev ? gdOf(f) - gdOf(prev.w) : 0,
      ...Object.fromEntries(ROLES.map(role => [`gold_${role}`, ((pb[role]?.totalGold || 0) - (pr[role]?.totalGold || 0)) / 1000])),
      levels: lv(b) - lv(r), elo: eloDiff / 100,
    };
  }

  function evalV2(x) {
    const v = model.features.map(k => x[k] ?? 0);
    if (model.type === 'linear') return sigmoid(model.bias + model.weights.reduce((a, w, i) => a + w * v[i], 0));
    let z = 0;
    for (const t of model.trees) {
      let n = t.root;
      while (n >= 0) n = v[t.f[n]] <= t.t[n] ? t.l[n] : t.r[n];
      z += t.v[~n];
    }
    return sigmoid(z);
  }

  // 對局 S 第 i 筆快照的藍方勝率
  function probAt(S, i) {
    const eloDiff = eloOf(S.g.teams.blue) - eloOf(S.g.teams.red);
    if (model.version !== 2) return probV1(S.snaps[i].w, S.snaps[i].gt / 1000, S.g.teams);
    const roleOf = Object.fromEntries(Object.entries(S.players).map(([id, p]) => [id, p.role]));
    return evalV2(featuresV2(S.snaps, i, roleOf, eloDiff));
  }

  function pregame(teams) {
    const eloDiff = eloOf(teams.blue) - eloOf(teams.red);
    if (model.version !== 2) {
      const empty = { totalGold: 0, totalKills: 0, towers: 0, dragons: [], barons: 0, inhibitors: 0 };
      return probV1({ blueTeam: empty, redTeam: empty }, 0, teams);
    }
    return evalV2({ elo: eloDiff / 100 });
  }

  return { load, probAt, pregame, elo: t => model ? Math.round(eloOf(t)) : null, get model() { return model; } };
})();

const pctWP = p => (p * 100).toFixed(0) + '%';
function wpBar(p, T, title = '即時勝率') {
  return `<div class="wpbar"><div class="lbl"><span><b style="color:var(--blue)">${pctWP(p)}</b> ${esc(T.blue.code)}</span>
    <span class="muted">${title}</span><span>${esc(T.red.code)} <b style="color:var(--red)">${pctWP(1 - p)}</b></span></div>
    <div class="track"><i class="b" style="width:${p * 100}%"></i><i class="r" style="width:${(1 - p) * 100}%"></i></div></div>`;
}
