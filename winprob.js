// 即時勝率：讀取 build_winprob.py 訓練出的邏輯斯迴歸權重，特徵定義需與 Python 端 features() 一致
const WP = (() => {
  let model = null;

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

  function features(f, tSec, eloDiff) {
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

  // 藍方勝率
  function prob(frame, tSec, teams) {
    const x = features(frame, tSec, eloOf(teams.blue) - eloOf(teams.red));
    return sigmoid(model.features.reduce((s, k, i) => s + model.weights[i] * x[k], 0));
  }

  const empty = { totalGold: 0, totalKills: 0, towers: 0, dragons: [], barons: 0, inhibitors: 0 };
  const pregame = teams => prob({ blueTeam: empty, redTeam: empty }, 0, teams);

  return { load, prob, pregame, elo: t => model ? Math.round(eloOf(t)) : null, get model() { return model; } };
})();

const pctWP = p => (p * 100).toFixed(0) + '%';
function wpBar(p, T, title = '即時勝率') {
  return `<div class="wpbar"><div class="lbl"><span><b style="color:var(--blue)">${pctWP(p)}</b> ${esc(T.blue.code)}</span>
    <span class="muted">${title}</span><span>${esc(T.red.code)} <b style="color:var(--red)">${pctWP(1 - p)}</b></span></div>
    <div class="track"><i class="b" style="width:${p * 100}%"></i><i class="r" style="width:${(1 - p) * 100}%"></i></div></div>`;
}
