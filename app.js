'use strict';
/* =====================================================================
   RNN State Traversal Tracker
   Theory of Computation – TAE

   Idea: a DFA has a finite set of discrete states. A trained RNN keeps a
   continuous hidden vector h_t. We run both machines on the same input and
   PROJECT every h_t onto a DFA state class in two ways:
     1. Readout  – the RNN's own softmax layer (learned classifier)
     2. Centroid – nearest class-mean of hidden states (geometric clustering)
   Then we compare the projected path with the true DFA path.
   ===================================================================== */

/* ---------------------------- utilities ---------------------------- */
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function randn(rng) {
  let u = 0;
  while (u === 0) u = rng();
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function argmax(arr) {
  let b = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[b]) b = i;
  return b;
}
function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/* ------------------------- DFA definitions ------------------------- */
// delta[state][symbolIndex] = nextState
const PALETTE = ['#2f4cdd', '#d98200', '#0f9d8a', '#c2338a'];
const DEAD = '#6b7280';

const DFAS = {
  parity: {
    name: 'Even number of 1s',
    alphabet: ['0', '1'],
    states: ['E', 'O'],
    colors: [PALETTE[0], PALETTE[1]],
    start: 0, accept: [0],
    delta: [[0, 1], [1, 0]],
    desc: 'Accepts binary strings that contain an even count of 1s. State E = even so far, O = odd so far.'
  },
  mod3: {
    name: 'Binary number divisible by 3',
    alphabet: ['0', '1'],
    states: ['r0', 'r1', 'r2'],
    colors: [PALETTE[0], PALETTE[1], PALETTE[2]],
    start: 0, accept: [0],
    delta: [[0, 1], [2, 0], [1, 2]],
    desc: 'Reads a binary number MSB-first and tracks its remainder mod 3. Accepts when the remainder is 0.'
  },
  no11: {
    name: 'No two consecutive 1s',
    alphabet: ['0', '1'],
    states: ['q0', 'q1', 'dead'],
    colors: [PALETTE[0], PALETTE[1], DEAD],
    start: 0, accept: [0, 1],
    delta: [[0, 1], [0, 2], [2, 2]],
    desc: 'Accepts strings that never contain "11". q0 = last symbol was not 1, q1 = last symbol was 1, dead = "11" seen (absorbing).'
  },
  ends01: {
    name: 'Ends with 01',
    alphabet: ['0', '1'],
    states: ['s0', 's1', 's2'],
    colors: [PALETTE[0], PALETTE[1], PALETTE[2]],
    start: 0, accept: [2],
    delta: [[1, 0], [1, 2], [1, 0]],
    desc: 'Accepts strings whose last two symbols are 0 then 1. s1 = just read 0, s2 = just read 01.'
  }
};

// Test inputs: a mix of in-language, not-in-language, short, long and malformed.
const TEST_STRINGS = [
  '', '0', '1', '00', '01', '10', '11', '101', '110', '111', '1001', '1010', '1100',
  '10101', '100100', '111111', '1011011', '10101010', '1100110011', '0000000000',
  '1111111111', '10011010110100101101',
  '110101110010110101101001110100',
  '1011010011101010010110110100101101001011',
  '10a1', '2', '1 0', '0b101', 'abc', '10-1', '１０'
];

function dfaRun(dfa, seq) {
  const path = [dfa.start];
  let s = dfa.start;
  for (const x of seq) { s = dfa.delta[s][x]; path.push(s); }
  return path;
}
function toSeq(dfa, str) { return Array.from(str).map(c => dfa.alphabet.indexOf(c)); }
function badSymbols(dfa, str) {
  return [...new Set(Array.from(str).filter(c => !dfa.alphabet.includes(c)))];
}
function classify(dfa, str) {
  const bad = badSymbols(dfa, str);
  if (bad.length) return { kind: 'malformed', bad };
  const path = dfaRun(dfa, toSeq(dfa, str));
  return { kind: dfa.accept.includes(path[path.length - 1]) ? 'accepted' : 'rejected' };
}
function randomSeq(rng, A, minLen, maxLen) {
  const len = minLen + Math.floor(rng() * (maxLen - minLen + 1));
  const s = new Array(len);
  for (let i = 0; i < len; i++) s[i] = Math.floor(rng() * A);
  return s;
}

/* ----------------------------- the RNN ----------------------------- */
// h_t = tanh(Wxh x_t + Whh h_{t-1} + bh)      (Elman RNN, h_0 = 0)
// y_t = softmax(Why h_t + by)                  (one score per DFA state)
class RNN {
  constructor(A, H, Q, seed) {
    this.A = A; this.H = H; this.Q = Q;
    const rng = mulberry32(seed);
    const mk = (n, scale) => {
      const w = new Float64Array(n);
      for (let i = 0; i < n; i++) w[i] = randn(rng) * scale;
      return { w, g: new Float64Array(n), m: new Float64Array(n), v: new Float64Array(n) };
    };
    this.Wxh = mk(H * A, 0.6);
    this.Whh = mk(H * H, 1 / Math.sqrt(H));
    this.bh = mk(H, 0);
    this.Why = mk(Q * H, 0.3);
    this.by = mk(Q, 0);
    this.params = [this.Wxh, this.Whh, this.bh, this.Why, this.by];
    this.t = 0;
  }

  readoutProbs(h) {
    const { H, Q } = this;
    const z = new Float64Array(Q);
    let mx = -Infinity;
    for (let q = 0; q < Q; q++) {
      let s = this.by.w[q];
      for (let j = 0; j < H; j++) s += this.Why.w[q * H + j] * h[j];
      z[q] = s; if (s > mx) mx = s;
    }
    let sum = 0;
    for (let q = 0; q < Q; q++) { z[q] = Math.exp(z[q] - mx); sum += z[q]; }
    for (let q = 0; q < Q; q++) z[q] /= sum;
    return z;
  }

  forward(seq) {
    const { A, H } = this;
    const hs = [new Float64Array(H)];
    const probs = [this.readoutProbs(hs[0])];
    for (let k = 0; k < seq.length; k++) {
      const prev = hs[k], x = seq[k], h = new Float64Array(H);
      for (let j = 0; j < H; j++) {
        let a = this.Wxh.w[j * A + x] + this.bh.w[j];
        const row = j * H;
        for (let i = 0; i < H; i++) a += this.Whh.w[row + i] * prev[i];
        h[j] = Math.tanh(a);
      }
      hs.push(h);
      probs.push(this.readoutProbs(h));
    }
    return { hs, probs };
  }

  // Back-propagation through time for one sequence (adds into .g)
  backward(seq, fw, targets) {
    const { A, H, Q } = this;
    let dhNext = new Float64Array(H);
    for (let k = seq.length; k >= 0; k--) {
      const h = fw.hs[k];
      const dl = Float64Array.from(fw.probs[k]);
      dl[targets[k]] -= 1;
      const dh = new Float64Array(H);
      for (let q = 0; q < Q; q++) {
        this.by.g[q] += dl[q];
        for (let j = 0; j < H; j++) {
          this.Why.g[q * H + j] += dl[q] * h[j];
          dh[j] += this.Why.w[q * H + j] * dl[q];
        }
      }
      if (k === 0) break;
      const x = seq[k - 1], hp = fw.hs[k - 1];
      const da = new Float64Array(H);
      for (let j = 0; j < H; j++) da[j] = (dh[j] + dhNext[j]) * (1 - h[j] * h[j]);
      const nxt = new Float64Array(H);
      for (let j = 0; j < H; j++) {
        this.Wxh.g[j * A + x] += da[j];
        this.bh.g[j] += da[j];
        for (let i = 0; i < H; i++) {
          this.Whh.g[j * H + i] += da[j] * hp[i];
          nxt[i] += this.Whh.w[j * H + i] * da[j];
        }
      }
      dhNext = nxt;
    }
  }

  trainBatch(batch, dfa, lr) {
    for (const p of this.params) p.g.fill(0);
    let loss = 0, n = 0, correct = 0;
    for (const seq of batch) {
      const fw = this.forward(seq);
      const tg = dfaRun(dfa, seq);
      for (let k = 0; k <= seq.length; k++) {
        loss += -Math.log(Math.max(fw.probs[k][tg[k]], 1e-12));
        if (argmax(fw.probs[k]) === tg[k]) correct++;
        n++;
      }
      this.backward(seq, fw, tg);
    }
    let norm = 0;
    for (const p of this.params) for (let i = 0; i < p.g.length; i++) { p.g[i] /= n; norm += p.g[i] * p.g[i]; }
    norm = Math.sqrt(norm);
    const scale = norm > 5 ? 5 / norm : 1;
    this.t++;
    const b1 = 0.9, b2 = 0.999;
    const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
    for (const p of this.params) {
      for (let i = 0; i < p.w.length; i++) {
        const g = p.g[i] * scale;
        p.m[i] = b1 * p.m[i] + (1 - b1) * g;
        p.v[i] = b2 * p.v[i] + (1 - b2) * g * g;
        p.w[i] -= lr * (p.m[i] / c1) / (Math.sqrt(p.v[i] / c2) + 1e-8);
      }
    }
    return { loss: loss / n, acc: correct / n };
  }
}

/* ------------------- projection assets (centroids, PCA) ------------------- */
function buildAssets(model, dfa, maxLen, seed) {
  const { A, H, Q } = model;
  const rng = mulberry32(seed + 999);
  const sums = Array.from({ length: Q }, () => new Float64Array(H));
  const counts = new Array(Q).fill(0);
  const states = [];   // all hidden vectors
  const labels = [];   // true DFA state for each
  for (let n = 0; n < 500; n++) {
    const seq = randomSeq(rng, A, 1, maxLen);
    const fw = model.forward(seq);
    const tg = dfaRun(dfa, seq);
    for (let k = 0; k <= seq.length; k++) {
      states.push(fw.hs[k]); labels.push(tg[k]);
      counts[tg[k]]++;
      for (let j = 0; j < H; j++) sums[tg[k]][j] += fw.hs[k][j];
    }
  }
  const centroids = sums.map((s, q) => counts[q] ? s.map(v => v / counts[q]) : null);

  // PCA by power iteration on the covariance matrix
  const mean = new Float64Array(H);
  for (const h of states) for (let j = 0; j < H; j++) mean[j] += h[j] / states.length;
  const C = Array.from({ length: H }, () => new Float64Array(H));
  for (const h of states) {
    for (let i = 0; i < H; i++) {
      const di = h[i] - mean[i];
      for (let j = 0; j < H; j++) C[i][j] += di * (h[j] - mean[j]) / states.length;
    }
  }
  let trace = 0; for (let i = 0; i < H; i++) trace += C[i][i];
  const comps = [], vars = [];
  for (let c = 0; c < 2; c++) {
    let v = new Float64Array(H);
    for (let i = 0; i < H; i++) v[i] = 1 + 0.13 * i + 0.37 * c;
    for (let it = 0; it < 300; it++) {
      const w = new Float64Array(H);
      for (let i = 0; i < H; i++) { let s = 0; for (let j = 0; j < H; j++) s += C[i][j] * v[j]; w[i] = s; }
      let nrm = 0; for (let i = 0; i < H; i++) nrm += w[i] * w[i];
      nrm = Math.sqrt(nrm) || 1;
      for (let i = 0; i < H; i++) v[i] = w[i] / nrm;
    }
    let lam = 0;
    for (let i = 0; i < H; i++) { let s = 0; for (let j = 0; j < H; j++) s += C[i][j] * v[j]; lam += v[i] * s; }
    for (let i = 0; i < H; i++) for (let j = 0; j < H; j++) C[i][j] -= lam * v[i] * v[j];
    comps.push(v); vars.push(trace > 0 ? lam / trace : 0);
  }
  const pca = {
    mean, comps, vars,
    project(h) {
      let x = 0, y = 0;
      for (let j = 0; j < H; j++) { const d = h[j] - mean[j]; x += d * comps[0][j]; y += d * comps[1][j]; }
      return [x, y];
    }
  };

  // sub-sample of points to draw as the "state cloud"
  const cloud = [];
  const step = Math.max(1, Math.floor(states.length / 700));
  for (let i = 0; i < states.length; i += step) cloud.push({ p: pca.project(states[i]), s: labels[i] });
  const cents2d = centroids.map(c => (c ? pca.project(c) : null));
  return { centroids, pca, cloud, cents2d };
}

function nearestCentroid(assets, h) {
  let best = -1, bd = Infinity;
  assets.centroids.forEach((c, q) => {
    if (!c) return;
    let d = 0;
    for (let j = 0; j < h.length; j++) { const e = h[j] - c[j]; d += e * e; }
    if (d < bd) { bd = d; best = q; }
  });
  return { state: best, dist: Math.sqrt(bd) };
}

/* ------------------------------ analysis ------------------------------ */
function analyze(model, dfa, assets, str) {
  const bad = badSymbols(dfa, str);
  if (bad.length) return { ok: false, bad, input: str };
  const seq = toSeq(dfa, str);
  const fw = model.forward(seq);
  const truth = dfaRun(dfa, seq);
  const steps = [];
  let matchR = 0, matchC = 0;
  for (let k = 0; k <= seq.length; k++) {
    const probs = fw.probs[k];
    const r = argmax(probs);
    const c = nearestCentroid(assets, fw.hs[k]);
    if (r === truth[k]) matchR++;
    if (c.state === truth[k]) matchC++;
    steps.push({
      k, sym: k === 0 ? 'ε' : dfa.alphabet[seq[k - 1]],
      h: fw.hs[k], pc: assets.pca.project(fw.hs[k]),
      dfa: truth[k], readout: r, conf: probs[r], centroid: c.state, cdist: c.dist
    });
  }
  const last = steps[steps.length - 1];
  const acc = s => dfa.accept.includes(s);
  return {
    ok: true, input: str, steps,
    dfaAccept: acc(last.dfa), rnnAccept: acc(last.readout), centAccept: acc(last.centroid),
    matchR, matchC, total: steps.length
  };
}

function evaluate(model, dfa, assets, maxLen, seed) {
  const rng = mulberry32(seed + 4242);
  const Q = dfa.states.length;
  const confusion = Array.from({ length: Q }, () => new Array(Q).fill(0));
  const mk = () => ({ n: 0, r: 0, c: 0, strings: 0, fin: 0 });
  const res = { in: mk(), long: mk() };
  for (const grp of ['in', 'long']) {
    for (let i = 0; i < 250; i++) {
      const seq = grp === 'in' ? randomSeq(rng, model.A, 1, maxLen) : randomSeq(rng, model.A, maxLen + 1, maxLen * 2);
      const fw = model.forward(seq);
      const tg = dfaRun(dfa, seq);
      for (let k = 0; k <= seq.length; k++) {
        const r = argmax(fw.probs[k]);
        const c = nearestCentroid(assets, fw.hs[k]).state;
        confusion[tg[k]][r]++;
        res[grp].n++;
        if (r === tg[k]) res[grp].r++;
        if (c === tg[k]) res[grp].c++;
      }
      res[grp].strings++;
      const last = seq.length;
      if (dfa.accept.includes(argmax(fw.probs[last])) === dfa.accept.includes(tg[last])) res[grp].fin++;
    }
  }
  return { res, confusion };
}

/* -------------------- export for node-based testing -------------------- */
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { DFAS, RNN, mulberry32, randomSeq, dfaRun, buildAssets, evaluate, analyze, classify, TEST_STRINGS };
}

/* =====================================================================
   UI
   ===================================================================== */
if (typeof document !== 'undefined') {
  (function () {
    const $ = id => document.getElementById(id);
    const NS = 'http://www.w3.org/2000/svg';
    const INK = '#14213d', GRID = '#d3dae3', PAPER = '#f7f8fa', COBALT = '#2f4cdd', AMBER = '#d98200';

    const app = {
      key: 'parity', dfa: DFAS.parity, model: null, assets: null, metrics: null,
      trace: null, step: 0, lossHist: [], trainToken: 0, playTimer: null, trained: false, deferredInstall: null
    };

    /* ---------- small helpers ---------- */
    function chip(q, extra) {
      const d = app.dfa;
      return `<span class="chip" style="--c:${d.colors[q]}">${esc(d.states[q])}${extra || ''}</span>`;
    }
    function verdictWord(b) { return b ? 'Accept' : 'Reject'; }
    function setupCanvas(c) {
      const dpr = window.devicePixelRatio || 1;
      const w = c.clientWidth, h = c.clientHeight;
      if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
        c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
      }
      const ctx = c.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      return { ctx, w, h };
    }

    /* ---------- DFA diagram (SVG) ---------- */
    function el(name, attrs, parent) {
      const e = document.createElementNS(NS, name);
      for (const k in attrs) e.setAttribute(k, attrs[k]);
      if (parent) parent.appendChild(e);
      return e;
    }
    function unit(x, y) { const l = Math.hypot(x, y) || 1; return [x / l, y / l]; }

    function renderDFA() {
      const dfa = app.dfa, svg = $('dfaSvg'), n = dfa.states.length;
      svg.innerHTML = '';
      const R = 22, cx = 220, cy = 170;
      const defs = el('defs', {}, svg);
      [['arrow', '#7d8794'], ['arrowA', INK]].forEach(([id, col]) => {
        const m = el('marker', { id, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
        el('path', { d: 'M0,0 L10,5 L0,10 z', fill: col }, m);
      });
      const pos = dfa.states.map((_, i) => {
        if (n === 1) return [cx, cy];
        if (n === 2) return [cx + (i === 0 ? -100 : 100), cy];
        const a = -Math.PI / 2 + (2 * Math.PI * i) / n;
        return [cx + 95 * Math.cos(a), cy + 95 * Math.sin(a)];
      });
      const outward = i => (n <= 2 ? [0, -1] : unit(pos[i][0] - cx, pos[i][1] - cy));

      // group edges by (from,to)
      const groups = {};
      dfa.delta.forEach((row, i) => row.forEach((j, x) => {
        const key = i + '-' + j;
        (groups[key] = groups[key] || { i, j, syms: [] }).syms.push(dfa.alphabet[x]);
      }));
      const edgeEls = {};
      Object.values(groups).forEach(g => {
        const gEl = el('g', { class: 'edge' }, svg);
        const [pi, pj] = [pos[g.i], pos[g.j]];
        let path, lx, ly;
        if (g.i === g.j) {
          const o = outward(g.i), t = [-o[1], o[0]];
          const s = unit(o[0] - 0.55 * t[0], o[1] - 0.55 * t[1]);
          const e = unit(o[0] + 0.55 * t[0], o[1] + 0.55 * t[1]);
          const S = [pi[0] + s[0] * R, pi[1] + s[1] * R], E = [pi[0] + e[0] * (R + 5), pi[1] + e[1] * (R + 5)];
          const c1 = [pi[0] + o[0] * (R + 54) - t[0] * 38, pi[1] + o[1] * (R + 54) - t[1] * 38];
          const c2 = [pi[0] + o[0] * (R + 54) + t[0] * 38, pi[1] + o[1] * (R + 54) + t[1] * 38];
          path = `M${S[0]},${S[1]} C${c1[0]},${c1[1]} ${c2[0]},${c2[1]} ${E[0]},${E[1]}`;
          lx = pi[0] + o[0] * (R + 50); ly = pi[1] + o[1] * (R + 50);
        } else {
          const dx = pj[0] - pi[0], dy = pj[1] - pi[1], len = Math.hypot(dx, dy);
          const nx = -dy / len, ny = dx / len;
          const c = [(pi[0] + pj[0]) / 2 + nx * 30, (pi[1] + pj[1]) / 2 + ny * 30];
          const su = unit(c[0] - pi[0], c[1] - pi[1]), eu = unit(c[0] - pj[0], c[1] - pj[1]);
          const S = [pi[0] + su[0] * R, pi[1] + su[1] * R], E = [pj[0] + eu[0] * (R + 5), pj[1] + eu[1] * (R + 5)];
          path = `M${S[0]},${S[1]} Q${c[0]},${c[1]} ${E[0]},${E[1]}`;
          lx = 0.25 * S[0] + 0.5 * c[0] + 0.25 * E[0] + nx * 10;
          ly = 0.25 * S[1] + 0.5 * c[1] + 0.25 * E[1] + ny * 10;
        }
        const p = el('path', { d: path, fill: 'none', stroke: '#7d8794', 'stroke-width': 1.6, 'marker-end': 'url(#arrow)' }, gEl);
        const t = el('text', { x: lx, y: ly + 4, 'text-anchor': 'middle', class: 'edge-label' }, gEl);
        t.textContent = g.syms.join(',');
        edgeEls[g.i + '-' + g.j] = { p, t };
      });

      // start arrow
      const s0 = pos[dfa.start];
      el('line', { x1: s0[0] - R - 44, y1: s0[1], x2: s0[0] - R - 5, y2: s0[1], stroke: INK, 'stroke-width': 1.8, 'marker-end': 'url(#arrowA)' }, svg);

      // nodes
      const nodeEls = dfa.states.map((name, i) => {
        const g = el('g', { class: 'node' }, svg);
        if (dfa.accept.includes(i)) el('circle', { cx: pos[i][0], cy: pos[i][1], r: R + 5, fill: 'none', stroke: dfa.colors[i], 'stroke-width': 1.6 }, g);
        const c = el('circle', { cx: pos[i][0], cy: pos[i][1], r: R, fill: '#fff', stroke: dfa.colors[i], 'stroke-width': 2.6 }, g);
        const t = el('text', { x: pos[i][0], y: pos[i][1] + 4.5, 'text-anchor': 'middle', class: 'node-label', fill: INK }, g);
        t.textContent = name;
        return { c, t };
      });
      svg._nodes = nodeEls; svg._edges = edgeEls;

      $('dfaLegend').innerHTML =
        dfa.states.map((_, i) => chip(i, dfa.accept.includes(i) ? ' <small>accepting</small>' : '')).join('') +
        `<span class="muted">Start arrow enters ${esc(dfa.states[dfa.start])}.</span>`;
    }

    function highlightDFA() {
      const svg = $('dfaSvg'); if (!svg._nodes) return;
      const tr = app.trace;
      svg._nodes.forEach(n => { n.c.setAttribute('fill', '#fff'); n.t.setAttribute('fill', INK); });
      Object.values(svg._edges).forEach(e => { e.p.setAttribute('stroke', '#7d8794'); e.p.setAttribute('stroke-width', 1.6); e.p.setAttribute('marker-end', 'url(#arrow)'); });
      if (!tr || !tr.ok) return;
      const st = tr.steps[app.step];
      svg._nodes[st.dfa].c.setAttribute('fill', app.dfa.colors[st.dfa]);
      svg._nodes[st.dfa].t.setAttribute('fill', '#fff');
      if (app.step > 0) {
        const e = svg._edges[tr.steps[app.step - 1].dfa + '-' + st.dfa];
        if (e) { e.p.setAttribute('stroke', INK); e.p.setAttribute('stroke-width', 3); e.p.setAttribute('marker-end', 'url(#arrowA)'); }
      }
    }

    /* ---------- PCA scatter ---------- */
    function drawPCA() {
      const { ctx, w, h } = setupCanvas($('pcaCanvas'));
      const A = app.assets;
      if (!A) return;
      const tr = app.trace && app.trace.ok ? app.trace : null;
      let pts = A.cloud.map(c => c.p);
      if (tr) pts = pts.concat(tr.steps.map(s => s.pc));
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      pts.forEach(([x, y]) => { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); });
      const padx = (x1 - x0) * 0.1 || 1, pady = (y1 - y0) * 0.1 || 1;
      x0 -= padx; x1 += padx; y0 -= pady; y1 += pady;
      const m = 14;
      const X = x => m + ((x - x0) / (x1 - x0)) * (w - 2 * m);
      const Y = y => h - m - ((y - y0) / (y1 - y0)) * (h - 2 * m);

      // grid
      ctx.strokeStyle = GRID; ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const gx = m + (i / 4) * (w - 2 * m), gy = m + (i / 4) * (h - 2 * m);
        ctx.beginPath(); ctx.moveTo(gx, m); ctx.lineTo(gx, h - m); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(m, gy); ctx.lineTo(w - m, gy); ctx.stroke();
      }
      // cloud
      A.cloud.forEach(c => {
        ctx.fillStyle = app.dfa.colors[c.s] + '55';
        ctx.beginPath(); ctx.arc(X(c.p[0]), Y(c.p[1]), 3, 0, 7); ctx.fill();
      });
      // centroids
      A.cents2d.forEach((p, q) => {
        if (!p) return;
        ctx.strokeStyle = app.dfa.colors[q]; ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 11, 0, 7); ctx.stroke();
        ctx.fillStyle = app.dfa.colors[q]; ctx.font = '600 12px ui-monospace, Menlo, Consolas, monospace';
        ctx.fillText(app.dfa.states[q], X(p[0]) + 14, Y(p[1]) - 10);
      });
      // trajectory
      if (tr) {
        const upto = app.step;
        ctx.lineWidth = 2; ctx.strokeStyle = INK;
        for (let k = 1; k <= upto; k++) {
          const a = tr.steps[k - 1].pc, b = tr.steps[k].pc;
          ctx.globalAlpha = 0.35 + 0.65 * (k / Math.max(upto, 1));
          ctx.beginPath(); ctx.moveTo(X(a[0]), Y(a[1])); ctx.lineTo(X(b[0]), Y(b[1])); ctx.stroke();
          // arrow head
          const ang = Math.atan2(Y(b[1]) - Y(a[1]), X(b[0]) - X(a[0]));
          const mx = (X(a[0]) + X(b[0])) / 2, my = (Y(a[1]) + Y(b[1])) / 2;
          ctx.beginPath(); ctx.moveTo(mx + 6 * Math.cos(ang), my + 6 * Math.sin(ang));
          ctx.lineTo(mx - 5 * Math.cos(ang - 0.6), my - 5 * Math.sin(ang - 0.6));
          ctx.lineTo(mx - 5 * Math.cos(ang + 0.6), my - 5 * Math.sin(ang + 0.6)); ctx.closePath();
          ctx.fillStyle = INK; ctx.fill();
        }
        ctx.globalAlpha = 1;
        for (let k = 0; k <= upto; k++) {
          const s = tr.steps[k], px = X(s.pc[0]), py = Y(s.pc[1]);
          ctx.fillStyle = '#fff'; ctx.strokeStyle = app.dfa.colors[s.dfa]; ctx.lineWidth = 2.5;
          const r = k === upto ? 8 : 5;
          ctx.beginPath(); ctx.arc(px, py, r, 0, 7); ctx.fill(); ctx.stroke();
          if (k === upto) { ctx.fillStyle = app.dfa.colors[s.dfa]; ctx.beginPath(); ctx.arc(px, py, 4, 0, 7); ctx.fill(); }
          if (tr.steps.length <= 24 || k === upto || k === 0) {
            ctx.fillStyle = INK; ctx.font = '11px ui-monospace, Menlo, Consolas, monospace';
            ctx.fillText('t' + k, px + 9, py + 14);
          }
        }
      }
      $('pcaNote').textContent = `PC1 ${(A.pca.vars[0] * 100).toFixed(0)}% + PC2 ${(A.pca.vars[1] * 100).toFixed(0)}% of hidden-state variance`;
    }

    /* ---------- hidden-state heatmap ---------- */
    function lerp(a, b, t) { return a + (b - a) * t; }
    function heatColor(v) {
      const base = [247, 248, 250], pos = [217, 130, 0], neg = [47, 76, 221];
      const tgt = v >= 0 ? pos : neg, t = Math.min(1, Math.abs(v));
      return `rgb(${Math.round(lerp(base[0], tgt[0], t))},${Math.round(lerp(base[1], tgt[1], t))},${Math.round(lerp(base[2], tgt[2], t))})`;
    }
    function drawHeat() {
      const { ctx, w, h } = setupCanvas($('heatCanvas'));
      const tr = app.trace;
      if (!tr || !tr.ok || !app.model) return;
      const H = app.model.H, n = tr.steps.length;
      const left = 30, bottom = 20, top = 4, right = 6;
      const cw = (w - left - right) / n, ch = (h - bottom - top) / H;
      ctx.font = '10px ui-monospace, Menlo, Consolas, monospace'; ctx.fillStyle = '#5b6575';
      for (let j = 0; j < H; j++) ctx.fillText('h' + j, 2, top + j * ch + ch / 2 + 3);
      for (let k = 0; k < n; k++) {
        for (let j = 0; j < H; j++) {
          ctx.fillStyle = heatColor(tr.steps[k].h[j]);
          ctx.fillRect(left + k * cw, top + j * ch, Math.max(cw - 1, 1), Math.max(ch - 1, 1));
        }
        if (cw >= 14 || k % Math.ceil(14 / cw) === 0) {
          ctx.fillStyle = '#5b6575'; ctx.textAlign = 'center';
          ctx.fillText(tr.steps[k].sym, left + k * cw + cw / 2, h - 6);
          ctx.textAlign = 'left';
        }
      }
      ctx.strokeStyle = INK; ctx.lineWidth = 2;
      ctx.strokeRect(left + app.step * cw, top, Math.max(cw - 1, 1), H * ch - 1);
    }

    /* ---------- trace table + verdict ---------- */
    function renderVerdict() {
      const tr = app.trace, box = $('verdictBox');
      if (!tr) { box.className = 'verdict idle'; box.innerHTML = '<p>Train the network, then trace an input to see both machines side by side.</p>'; return; }
      if (!tr.ok) {
        box.className = 'verdict bad';
        box.innerHTML = `<p><strong>Input rejected before the RNN ran.</strong> Symbol${tr.bad.length > 1 ? 's' : ''} ${tr.bad.map(b => '“' + esc(b) + '”').join(', ')} not in the alphabet {${app.dfa.alphabet.join(', ')}}. A DFA can only read symbols from its alphabet, so the input is invalid.</p>`;
        return;
      }
      const agree = tr.dfaAccept === tr.rnnAccept;
      box.className = 'verdict ' + (agree && tr.matchR === tr.total ? 'good' : 'warn');
      box.innerHTML = `
        <div><span class="k">DFA</span><strong>${verdictWord(tr.dfaAccept)}</strong></div>
        <div><span class="k">RNN readout</span><strong>${verdictWord(tr.rnnAccept)}</strong></div>
        <div><span class="k">Centroid</span><strong>${verdictWord(tr.centAccept)}</strong></div>
        <div><span class="k">Path match</span><strong>${tr.matchR}/${tr.total}</strong></div>
        <p class="verdict-note">${agree ? 'The RNN reaches the same final verdict as the DFA.' : 'The RNN disagrees with the DFA on the final verdict.'}
        ${tr.matchR === tr.total ? ' Every step projects onto the correct DFA state.' : ' At least one step projects onto the wrong DFA state.'}</p>`;
    }

    function renderTable() {
      const tr = app.trace, body = $('traceBody');
      if (!tr || !tr.ok) { body.innerHTML = '<tr><td colspan="6" class="empty">No trace yet.</td></tr>'; return; }
      body.innerHTML = tr.steps.map(s => `
        <tr data-k="${s.k}" class="${s.k === app.step ? 'cur' : ''}">
          <td>${s.k}</td><td class="sym">${esc(s.sym)}</td>
          <td>${chip(s.dfa)}</td>
          <td>${chip(s.readout)} <small>${(s.conf * 100).toFixed(0)}%</small></td>
          <td>${chip(s.centroid)} <small>d=${s.cdist.toFixed(2)}</small></td>
          <td class="${s.readout === s.dfa ? 'ok' : 'no'}">${s.readout === s.dfa ? 'match' : 'differs'}</td>
        </tr>`).join('');
    }

    function setStep(k) {
      const tr = app.trace; if (!tr || !tr.ok) return;
      app.step = Math.max(0, Math.min(tr.steps.length - 1, k));
      $('stepRange').value = app.step;
      $('stepLabel').textContent = `t = ${app.step} of ${tr.steps.length - 1}`;
      document.querySelectorAll('#traceBody tr').forEach(r => r.classList.toggle('cur', +r.dataset.k === app.step));
      const cur = document.querySelector('#traceBody tr.cur');
      if (cur && cur.scrollIntoView) cur.scrollIntoView({ block: 'nearest' });
      highlightDFA(); drawPCA(); drawHeat();
    }

    function stopPlay() { if (app.playTimer) { clearInterval(app.playTimer); app.playTimer = null; $('stepPlay').textContent = 'Play'; } }
    function togglePlay() {
      if (app.playTimer) { stopPlay(); return; }
      const tr = app.trace; if (!tr || !tr.ok) return;
      if (app.step >= tr.steps.length - 1) setStep(0);
      $('stepPlay').textContent = 'Pause';
      app.playTimer = setInterval(() => {
        if (app.step >= tr.steps.length - 1) { stopPlay(); return; }
        setStep(app.step + 1);
      }, 650);
    }

    function runTrace(str) {
      stopPlay();
      if (!app.trained) { $('trainStatus').textContent = 'Train the RNN first.'; return; }
      str = str.trim();
      $('inputStr').value = str;
      app.trace = analyze(app.model, app.dfa, app.assets, str);
      app.step = 0;
      const ok = app.trace.ok;
      ['stepPrev', 'stepPlay', 'stepNext', 'stepRange'].forEach(id => $(id).disabled = !ok);
      if (ok) { $('stepRange').max = app.trace.steps.length - 1; }
      $('stepLabel').textContent = ok ? `t = 0 of ${app.trace.steps.length - 1}` : '';
      renderVerdict(); renderTable();
      if (ok) setStep(0); else { highlightDFA(); drawPCA(); drawHeat(); const { ctx, w, h } = setupCanvas($('heatCanvas')); }
    }

    /* ---------- example chips ---------- */
    function renderChips() {
      const groups = { accepted: [], rejected: [], malformed: [] };
      TEST_STRINGS.forEach(s => {
        const k = classify(app.dfa, s).kind;
        if (s.length > 12 && k !== 'malformed') return;
        groups[k].push(s);
      });
      const pick = (arr, n) => arr.filter(s => s !== '').slice(0, n);
      const mk = (cls, label, arr) => pick(arr, 3).map(s => `<button type="button" class="ex ${cls}" data-s="${esc(s)}" title="${label}">${esc(s)}</button>`).join('');
      $('chips').innerHTML =
        `<span class="chip-group-label">In language</span>${mk('acc', 'In language', groups.accepted)}` +
        `<span class="chip-group-label">Not in language</span>${mk('rej', 'Not in language', groups.rejected)}` +
        `<span class="chip-group-label">Malformed</span>${mk('mal', 'Malformed', groups.malformed.slice(0, 2).concat(['2']).slice(0, 2))}` +
        `<button type="button" class="ex" data-s="" title="Empty string (ε)">ε (empty)</button>`;
    }

    /* ---------- loss chart ---------- */
    function drawLoss() {
      const { ctx, w, h } = setupCanvas($('lossCanvas'));
      const L = app.lossHist; if (L.length < 2) return;
      const mx = Math.max(...L, 0.01), p = 4;
      ctx.strokeStyle = GRID; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(p, h - p); ctx.lineTo(w - p, h - p); ctx.stroke();
      ctx.strokeStyle = COBALT; ctx.lineWidth = 2; ctx.beginPath();
      L.forEach((v, i) => {
        const x = p + (i / (L.length - 1)) * (w - 2 * p), y = h - p - (v / mx) * (h - 2 * p);
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.stroke();
    }

    /* ---------- metrics + confusion ---------- */
    function renderMetrics() {
      const M = app.metrics; if (!M) return;
      const pct = (a, b) => (b ? (100 * a / b).toFixed(1) + '%' : '–');
      const r = M.res;
      $('metricsGrid').innerHTML = `
        <div class="stat"><b>${pct(r.in.r, r.in.n)}</b><span>Readout step accuracy<br><small>strings up to training length</small></span></div>
        <div class="stat"><b>${pct(r.in.c, r.in.n)}</b><span>Centroid step accuracy<br><small>strings up to training length</small></span></div>
        <div class="stat ${r.long.r / r.long.n < 0.95 ? 'warnstat' : ''}"><b>${pct(r.long.r, r.long.n)}</b><span>Readout step accuracy<br><small>strings up to 2× training length</small></span></div>
        <div class="stat"><b>${pct(r.in.fin + r.long.fin, r.in.strings + r.long.strings)}</b><span>Accept / reject agreement<br><small>500 random strings</small></span></div>`;
      const d = app.dfa, Q = d.states.length;
      let mx = 1; M.confusion.forEach(row => row.forEach(v => { mx = Math.max(mx, v); }));
      let html = '<table class="conf"><thead><tr><th><small>true ↓ / projected →</small></th>' + d.states.map((_, i) => `<th>${chip(i)}</th>`).join('') + '</tr></thead><tbody>';
      for (let i = 0; i < Q; i++) {
        html += `<tr><th>${chip(i)}</th>`;
        for (let j = 0; j < Q; j++) {
          const v = M.confusion[i][j], a = Math.min(1, v / mx);
          const col = i === j ? '47,76,221' : '194,51,66';
          html += `<td style="background:rgba(${col},${(a * 0.75).toFixed(2)});color:${a > 0.5 ? '#fff' : INK}">${v}</td>`;
        }
        html += '</tr>';
      }
      $('confusion').innerHTML = html + '</tbody></table>';
    }

    /* ---------- batch test suite ---------- */
    function runSuite() {
      if (!app.trained) return;
      let pass = 0;
      const rows = TEST_STRINGS.map(s => {
        const shown = s === '' ? 'ε (empty)' : s;
        const cls = classify(app.dfa, s);
        if (cls.kind === 'malformed') {
          pass++;
          return `<tr><td class="mono">${esc(shown)}</td><td><span class="tag mal">Malformed</span></td><td colspan="3" class="muted">Rejected by input check: ${cls.bad.map(b => '“' + esc(b) + '”').join(', ')} not in alphabet</td><td class="ok">PASS</td></tr>`;
        }
        const a = analyze(app.model, app.dfa, app.assets, s);
        const good = a.dfaAccept === a.rnnAccept && a.matchR === a.total;
        if (good) pass++;
        return `<tr data-s="${esc(s)}" class="clickable"><td class="mono">${esc(shown.length > 22 ? shown.slice(0, 20) + '…' : shown)}<small class="len"> len ${s.length}</small></td>
          <td><span class="tag ${cls.kind === 'accepted' ? 'acc' : 'rej'}">${cls.kind === 'accepted' ? 'In language' : 'Not in language'}</span></td>
          <td>${verdictWord(a.dfaAccept)}</td><td>${verdictWord(a.rnnAccept)}</td><td>${a.matchR}/${a.total}</td>
          <td class="${good ? 'ok' : 'no'}">${good ? 'PASS' : 'MISMATCH'}</td></tr>`;
      });
      $('batchBody').innerHTML = rows.join('');
      $('batchSummary').textContent = `${pass} of ${TEST_STRINGS.length} test inputs behave exactly like the DFA.`;
    }

    /* ---------- training ---------- */
    function startTraining() {
      stopPlay();
      const token = ++app.trainToken;
      app.key = $('dfaSelect').value; app.dfa = DFAS[app.key];
      const dfa = app.dfa;
      const H = +$('hiddenSize').value, total = +$('iters').value, maxLen = +$('maxLen').value, seed = +$('seed').value || 1;
      app.model = new RNN(dfa.alphabet.length, H, dfa.states.length, seed);
      app.trained = false; app.assets = null; app.trace = null; app.metrics = null; app.lossHist = [];
      $('trainBtn').disabled = true; $('trainBtn').textContent = 'Training…';
      ['runBtn', 'batchBtn'].forEach(id => $(id).disabled = true);
      renderDFA(); renderChips(); renderVerdict(); renderTable(); highlightDFA();
      $('metricsGrid').innerHTML = ''; $('confusion').innerHTML = ''; $('batchBody').innerHTML = ''; $('batchSummary').textContent = '';
      const rng = mulberry32(seed + 17);
      let it = 0, ema = null;
      function frame() {
        if (token !== app.trainToken) return;
        const t0 = performance.now();
        while (it < total && performance.now() - t0 < 24) {
          const batch = []; for (let b = 0; b < 32; b++) batch.push(randomSeq(rng, dfa.alphabet.length, 1, maxLen));
          const lr = 0.02 * (1 - 0.85 * it / total);
          const r = app.model.trainBatch(batch, dfa, lr);
          ema = ema === null ? r.loss : 0.95 * ema + 0.05 * r.loss;
          it++;
          if (it % 10 === 0) app.lossHist.push(ema);
          if (it > 300 && ema < 0.002) { it = total; break; }
        }
        $('progressBar').style.width = (100 * it / total) + '%';
        $('trainStatus').textContent = `Iteration ${it} / ${total} · loss ${ema.toFixed(4)}`;
        drawLoss();
        if (it < total) { requestAnimationFrame(frame); return; }
        finishTraining(ema, maxLen, seed);
      }
      requestAnimationFrame(frame);
    }

    function finishTraining(loss, maxLen, seed) {
      app.assets = buildAssets(app.model, app.dfa, maxLen, seed);
      app.metrics = evaluate(app.model, app.dfa, app.assets, maxLen, seed);
      app.trained = true;
      $('progressBar').style.width = '100%';
      $('trainStatus').textContent = `Trained · final loss ${loss.toFixed(4)} · ${app.model.H} hidden units`;
      $('trainBtn').disabled = false; $('trainBtn').textContent = 'Retrain';
      ['runBtn', 'batchBtn'].forEach(id => $(id).disabled = false);
      renderMetrics(); runSuite();
      runTrace($('inputStr').value || '1011');
    }

    /* ---------- install prompt ---------- */
    window.addEventListener('beforeinstallprompt', e => {
      e.preventDefault(); app.deferredInstall = e; $('installBtn').hidden = false;
    });
    window.addEventListener('appinstalled', () => { $('installBtn').hidden = true; });

    /* ---------- wiring ---------- */
    function init() {
      $('dfaSelect').innerHTML = Object.entries(DFAS).map(([k, d]) => `<option value="${k}">${esc(d.name)}</option>`).join('');
      const showDesc = () => {
        const d = DFAS[$('dfaSelect').value];
        $('dfaDesc').textContent = d.desc + ' Alphabet: {' + d.alphabet.join(', ') + '}.';
      };
      $('dfaSelect').addEventListener('change', () => {
        showDesc();
        app.key = $('dfaSelect').value; app.dfa = DFAS[app.key];
        renderDFA(); renderChips();
      });
      showDesc();
      $('trainBtn').addEventListener('click', startTraining);
      $('runBtn').addEventListener('click', () => runTrace($('inputStr').value));
      $('inputStr').addEventListener('keydown', e => { if (e.key === 'Enter') runTrace($('inputStr').value); });
      $('chips').addEventListener('click', e => { const b = e.target.closest('.ex'); if (b) runTrace(b.dataset.s); });
      $('stepPrev').addEventListener('click', () => { stopPlay(); setStep(app.step - 1); });
      $('stepNext').addEventListener('click', () => { stopPlay(); setStep(app.step + 1); });
      $('stepPlay').addEventListener('click', togglePlay);
      $('stepRange').addEventListener('input', e => { stopPlay(); setStep(+e.target.value); });
      $('traceBody').addEventListener('click', e => { const r = e.target.closest('tr[data-k]'); if (r) { stopPlay(); setStep(+r.dataset.k); } });
      $('batchBtn').addEventListener('click', runSuite);
      $('batchBody').addEventListener('click', e => {
        const r = e.target.closest('tr[data-s]');
        if (r) { runTrace(r.dataset.s); $('trace-section').scrollIntoView({ behavior: 'smooth' }); }
      });
      $('installBtn').addEventListener('click', async () => {
        if (!app.deferredInstall) return;
        app.deferredInstall.prompt(); await app.deferredInstall.userChoice;
        app.deferredInstall = null; $('installBtn').hidden = true;
      });
      window.addEventListener('resize', () => { drawPCA(); drawHeat(); drawLoss(); });

      if ('serviceWorker' in navigator && location.protocol !== 'file:') {
        navigator.serviceWorker.register('./sw.js').catch(() => { });
      }
      $('inputStr').value = '1011';
      renderDFA(); renderChips(); renderVerdict(); renderTable();
      startTraining();
    }
    document.addEventListener('DOMContentLoaded', init);
  })();
}
