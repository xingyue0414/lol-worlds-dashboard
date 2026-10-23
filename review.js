// 賽後復盤：比賽結束後，從整局快照（S.snaps）以規則推導輸方的主要敗因
const Review = (() => {
  const ROLE_ORDER = ['top', 'jungle', 'mid', 'bottom', 'support'];
  const other = s => s === 'blue' ? 'red' : 'blue';
  const team = (f, side) => f[side + 'Team'];

  // 從快照序列整理出結構化事件
  function timeline() {
    const deaths = [], objs = [];
    for (let i = 1; i < S.snaps.length; i++) {
      const a = S.snaps[i - 1], b = S.snaps[i];
      for (const side of ['blue', 'red']) {
        const pa = Object.fromEntries(team(a.w, side).participants.map(p => [p.participantId, p]));
        for (const p of team(b.w, side).participants)
          for (let k = 0; k < p.deaths - (pa[p.participantId]?.deaths ?? p.deaths); k++) deaths.push({ gt: b.gt, pid: p.participantId, side });
        const ta = team(a.w, side), tb = team(b.w, side);
        for (const d of tb.dragons.slice(ta.dragons.length)) objs.push({ gt: b.gt, side, kind: 'dragon', type: d, n: tb.dragons.length });
        if (tb.barons > ta.barons) objs.push({ gt: b.gt, side, kind: 'baron' });
        if (tb.towers > ta.towers) objs.push({ gt: b.gt, side, kind: 'tower', n: tb.towers - ta.towers });
        if (tb.inhibitors > ta.inhibitors) objs.push({ gt: b.gt, side, kind: 'inhib' });
      }
    }
    return { deaths, objs };
  }

  const snapAt = ms => S.snaps.find(s => s.gt >= ms) || S.snaps.at(-1);
  function roleState(snap) {
    const out = {};
    for (const side of ['blue', 'red'])
      for (const p of team(snap.w, side).participants) {
        const m = S.players[p.participantId];
        (out[m.role] ??= {})[side] = { ...p, name: m.name, champ: m.championId };
      }
    return out;
  }

  function analyze() {
    const last = S.snaps.at(-1), f = last.w;
    // 勝方：與訓練資料相同的判定（推塔 → 兵營 → 經濟）
    const key = s => [team(f, s).towers, team(f, s).inhibitors, team(f, s).totalGold];
    const kb = key('blue'), kr = key('red');
    const win = kb[0] !== kr[0] ? (kb[0] > kr[0] ? 'blue' : 'red') : kb[1] !== kr[1] ? (kb[1] > kr[1] ? 'blue' : 'red') : (kb[2] >= kr[2] ? 'blue' : 'red');
    const lose = other(win), W = S.g.teams[win].code, L = S.g.teams[lose].code;
    const { deaths, objs } = timeline();
    const reasons = []; // {score, html}
    const name = pid => esc(S.players[pid].name);
    // 「Meiko、Rookie、Meiko」→「Meiko×2、Rookie」
    const names = list => Object.entries(list.reduce((o, n) => (o[n] = (o[n] || 0) + 1, o), {})).map(([n, c]) => c > 1 ? `${n}×${c}` : n).join('、');
    const champ = c => esc(champName(c));
    const lp = p => lose === 'blue' ? p : 1 - p; // 輸方勝率

    // 1. 曾經領先卻沒能轉換
    let peak = null;
    for (const [gt, p] of S.wp) if (!peak || lp(p) > peak.p) peak = { gt, p: lp(p) };
    const objsBefore = (side, kind, t) => objs.filter(o => o.side === side && o.kind === kind && o.gt <= t).length;
    if (peak && peak.p >= 0.55) {
      const s = snapAt(peak.gt), tl = team(s.w, lose), tw = team(s.w, win);
      const gd = tl.totalGold - tw.totalGold;
      const dr = objsBefore(lose, 'dragon', peak.gt + 4 * 60000), drW = objsBefore(win, 'dragon', peak.gt + 4 * 60000);
      const tw4 = objsBefore(lose, 'tower', peak.gt + 4 * 60000);
      reasons.push({
        score: 60 + (peak.p - 0.5) * 100,
        html: `<b>領先沒有轉換成資源</b>：${L} 在 ${fmtT(peak.gt)} 勝率最高到 ${pctWP(peak.p)}（擊殺 ${tl.totalKills}:${tw.totalKills}、經濟 ${gd >= 0 ? '+' : ''}${fmtG(gd)}），` +
          `但接下來 4 分鐘只拿到 ${dr} 條小龍、${tw4} 座塔，同期 ${W} 拿了 ${drW} 條龍。`,
      });
    }

    // 2. 小龍 / 龍魂 / 巴龍控制
    const drW = objs.filter(o => o.side === win && o.kind === 'dragon'), drL = objs.filter(o => o.side === lose && o.kind === 'dragon');
    if (drW.length >= 2 && drW.length > drL.length) {
      const soul = drW.length >= 4 ? drW[3] : null;
      reasons.push({
        score: 40 + (drW.length - drL.length) * 10 + (soul ? 20 : 0),
        html: `<b>小龍控制 ${drW.length}:${drL.length}</b>：${W} 拿下 ${drW.map(o => `${DRAGON[o.type] || o.type}（${fmtT(o.gt)}）`).join('、')}` +
          (soul ? `，${fmtT(soul.gt)} 拿到<b>龍魂</b>` : '') + '。',
      });
    }
    const baron = objs.find(o => o.side === win && o.kind === 'baron');
    if (baron) {
      const tw = objs.filter(o => o.side === win && o.kind === 'tower' && o.gt > baron.gt && o.gt <= baron.gt + 180000).reduce((s, o) => s + o.n, 0);
      const toEnd = last.gt - baron.gt;
      reasons.push({
        score: 45 + (toEnd < 240000 ? 15 : 0),
        html: `<b>巴龍收尾</b>：${W} ${fmtT(baron.gt)} 拿下巴龍，之後 3 分鐘推掉 ${tw} 座塔` + (toEnd < 300000 ? `，${fmtT(toEnd)} 後結束比賽。` : '。'),
      });
    }

    // 3. 最大的轉折：輸方勝率 2 分鐘內跌最多的區段
    let drop = null;
    for (let i = 0; i < S.wp.length; i++) for (let j = i + 1; j < S.wp.length && S.wp[j][0] - S.wp[i][0] <= 120000; j++) {
      const d = lp(S.wp[i][1]) - lp(S.wp[j][1]);
      if (!drop || d > drop.d) drop = { d, from: S.wp[i], to: S.wp[j] };
    }
    if (drop && drop.d >= 0.15) {
      const [t1, t2] = [drop.from[0] - 30000, drop.to[0] + 10000];
      const dl = deaths.filter(x => x.gt >= t1 && x.gt <= t2 && x.side === lose), dw = deaths.filter(x => x.gt >= t1 && x.gt <= t2 && x.side === win);
      const ob = objs.filter(o => o.gt >= t1 && o.gt <= t2 && o.side === win && o.kind !== 'tower');
      reasons.push({
        score: 50 + drop.d * 60,
        html: `<b>關鍵轉折 ${fmtT(drop.from[0])}–${fmtT(drop.to[0])}</b>：${L} 勝率 ${pctWP(lp(drop.from[1]))} → ${pctWP(lp(drop.to[1]))}。` +
          `這段 ${L} 陣亡 ${dl.length} 人（${names(dl.map(x => name(x.pid))) || '—'}），${W} 只陣亡 ${dw.length} 人` +
          (ob.length ? `，並拿下${ob.map(o => o.kind === 'dragon' ? DRAGON[o.type] || '小龍' : o.kind === 'baron' ? '巴龍' : '水晶兵營').join('、')}` : '') + '。',
      });
    }

    // 4. 物件前陣亡（對手拿龍/巴龍前 60 秒內）
    // 每次陣亡只算給它之後最近的一個物件，避免同一波團戰被重複計算
    const objDeaths = [];
    const bigObjs = objs.filter(o => o.side === win && (o.kind === 'dragon' || o.kind === 'baron'));
    for (const d of deaths.filter(d => d.side === lose)) {
      const o = bigObjs.find(o => d.gt >= o.gt - 60000 && d.gt <= o.gt + 10000);
      if (o) objDeaths.push({ d, o });
    }
    if (objDeaths.length >= 2) {
      const byObj = {};
      for (const { d, o } of objDeaths) (byObj[o.gt] ??= { o, ns: [] }).ns.push(name(d.pid));
      reasons.push({
        score: 35 + objDeaths.length * 4,
        html: `<b>物件團戰前先掉人</b>：` + Object.values(byObj).map(({ o, ns }) =>
          `${fmtT(o.gt)} ${o.kind === 'baron' ? '巴龍' : DRAGON[o.type] || '小龍'}前 ${names(ns)} 陣亡`).join('；') + '，物件因此被對手白拿。',
      });
    }

    // 5. 各路經濟走勢：中期（14 分）到結束，輸方哪一路被拉開最多
    const mid = roleState(snapAt(14 * 60000)), end = roleState(last);
    const gaps = ROLE_ORDER.filter(r => end[r]?.[lose] && mid[r]?.[lose]).map(r => {
      const e = end[r], m = mid[r];
      return {
        r, a: e[lose], b: e[win],
        mid: m[lose].totalGold - m[win].totalGold, end: e[lose].totalGold - e[win].totalGold,
        csMid: m[lose].creepScore - m[win].creepScore, csEnd: e[lose].creepScore - e[win].creepScore,
      };
    });
    const worst = gaps.slice().sort((x, y) => x.end - y.end)[0];
    if (worst && worst.end <= -1500) {
      reasons.push({
        score: 30 + Math.min(40, -worst.end / 150),
        html: `<b>${ROLE[worst.r]}差距</b>：${esc(worst.a.name)}（${champ(worst.a.champ)}）對 ${esc(worst.b.name)}（${champ(worst.b.champ)}），` +
          `經濟從 14 分的 ${worst.mid >= 0 ? '+' : ''}${fmtG(worst.mid)} 被拉到結束時 ${fmtG(worst.end)}，CS ${worst.csEnd >= 0 ? '+' : ''}${worst.csEnd}；` +
          `${esc(worst.b.name)} 戰績 ${worst.b.kills}/${worst.b.deaths}/${worst.b.assists}。`,
      });
    }
    const leads = gaps.filter(g => g.mid >= 800);
    if (leads.length && worst?.end < 0) {
      reasons.push({
        score: 25 + leads.length * 5,
        html: `<b>線上優勢沒延續</b>：14 分時 ${leads.map(g => `${ROLE[g.r]} ${esc(g.a.name)} +${fmtG(g.mid)}`).join('、')}，` +
          `到結束變成 ${leads.map(g => `${g.end >= 0 ? '+' : ''}${fmtG(g.end)}`).join('、')}。`,
      });
    }

    // 6. 陣亡最多的選手、勝方 MVP
    const lastL = team(f, lose).participants.slice().sort((a, b) => b.deaths - a.deaths)[0];
    if (lastL && lastL.deaths >= 5)
      reasons.push({ score: 20 + lastL.deaths * 2, html: `<b>${name(lastL.participantId)} 陣亡 ${lastL.deaths} 次</b>，是 ${L} 送出最多擊殺的人。` });
    const det = Object.fromEntries((last.d || S.snaps.filter(s => s.d).at(-1)?.d)?.participants.map(p => [p.participantId, p]) || []);
    const mvp = team(f, win).participants.slice().sort((a, b) => (b.kills + b.assists) / Math.max(1, b.deaths) - (a.kills + a.assists) / Math.max(1, a.deaths))[0];
    if (mvp) {
      const x = det[mvp.participantId];
      reasons.push({
        score: 15,
        html: `<b>${W} 核心</b>：${name(mvp.participantId)}（${champ(S.players[mvp.participantId].championId)}）${mvp.kills}/${mvp.deaths}/${mvp.assists}` +
          (x ? `，傷害占比 ${Math.round(x.championDamageShare * 100)}%、參團 ${Math.round(x.killParticipation * 100)}%` : '') + '。',
      });
    }

    reasons.sort((a, b) => b.score - a.score);
    const tw = team(f, win), tl = team(f, lose);
    const summary = `<b class="${win}">${esc(W)}</b> 獲勝（${fmtT(last.gt)}）· 擊殺 ${tw.totalKills}:${tl.totalKills} · 塔 ${tw.towers}:${tl.towers} · 小龍 ${tw.dragons.length}:${tl.dragons.length} · 巴龍 ${tw.barons}:${tl.barons} · 經濟 +${fmtG(tw.totalGold - tl.totalGold)}`;
    return { win, lose, summary, reasons };
  }

  function render() {
    const box = $('#review');
    if (!S?.finished || S.snaps.length < 10) { box.hidden = true; return; }
    const { lose, summary, reasons } = analyze();
    box.hidden = false;
    box.innerHTML = `<h2>🔍 賽後復盤 · ${esc(S.g.teams[lose].code)} 為什麼輸</h2><div class="rv-sum">${summary}</div>
      <ol class="rv-list">${reasons.slice(0, 6).map(r => `<li>${r.html}</li>`).join('')}</ol>
      <div class="muted" style="font-size:12px">依比賽數據自動推導（勝率模型、物件時間、各路經濟），不含視野、技能與兵線細節。</div>`;
  }

  return { render, analyze };
})();
