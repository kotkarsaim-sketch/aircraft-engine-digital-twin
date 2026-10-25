/* =========================================================================
   AIRCRAFT ENGINE DIGITAL TWIN -- DASHBOARD FRONT-END
   -------------------------------------------------------------------------
   * Connects to backend.py over WebSocket (ws://localhost:8765)
   * Renders custom SVG gauges for the seven engine parameters
   * Animates a low-poly Cessna in Three.js (prop=RPM, cowl=EGT, anomaly=flash)
   * Fault injection controls with severity slider (bidirectional WS)
   * Health & Prognostics with explainability bars
   * Historical analytics charts (Chart.js) for sensors + AI metrics
   ========================================================================= */
"use strict";

/* ----------------------------- configuration --------------------------- */
const CFG = {
  WS_URL: "ws://" + (location.hostname || "localhost") + ":8765",
  RECONNECT_BASE_MS: 1500,
  RECONNECT_MAX_MS: 15000,
  MODEL_URL: null,
  PROP_VISUAL_DIV: 0.18,
  CHART_HISTORY: 120,               // 2 minutes of 1 Hz data
};

/* ------------------------------- utilities ----------------------------- */
const $ = (id) => document.getElementById(id);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

function fmtClock(ts) {
  const d = ts ? new Date(ts * 1000) : new Date();
  return d.toTimeString().slice(0, 8);
}

/* ------------------------------ event log ------------------------------ */
function logEvent(msg, cls) {
  const ul = $("log");
  if (!ul) return;
  const li = document.createElement("li");
  if (cls) li.className = cls;
  li.textContent = "[" + fmtClock() + "]  " + msg;
  ul.prepend(li);
  while (ul.children.length > 80) ul.removeChild(ul.lastChild);
}

/* =========================================================================
   SVG GAUGE WIDGET
   ========================================================================= */
class Gauge {
  constructor(def, mount) {
    this.def = def;
    this.min = def.min;
    this.max = def.max;
    this.cx = 100; this.cy = 92; this.r = 74;
    const NS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", "0 0 200 172");
    const el = (tag, attrs) => {
      const n = document.createElementNS(NS, tag);
      for (const k in attrs) n.setAttribute(k, attrs[k]);
      svg.appendChild(n);
      return n;
    };
    el("circle", { cx: this.cx, cy: this.cy, r: this.r + 12,
                   fill: "#0d1524", stroke: "#22304a" });
    let prev = this.min;
    for (const [limit, color] of def.zones) {
      el("path", { d: this.arcPath(prev, limit), fill: "none",
                   stroke: color, "stroke-width": 7 });
      prev = limit;
    }
    const step = (this.max - this.min) / 10;
    for (let i = 0; i <= 10; i++) {
      const v = this.min + i * step;
      const ang = this.valueToAngle(v);
      const [x1, y1] = this.polar(ang, this.r - 2);
      const [x2, y2] = this.polar(ang, this.r - (i % 2 === 0 ? 11 : 6));
      el("line", { x1, y1, x2, y2, stroke: "#33445f",
                   "stroke-width": i % 2 === 0 ? 1.6 : 1 });
      if (i % 5 === 0) {
        const [lx, ly] = this.polar(ang, this.r - 19);
        el("text", { x: lx, y: ly + 2.5, "text-anchor": "middle",
                     class: "tick-label" }).textContent = String(Math.round(v));
      }
    }
    this.needle = el("g", { class: "needle",
      transform: `rotate(${this.valueToAngle(this.min)} ${this.cx} ${this.cy})`});
    const path = document.createElementNS(NS, "path");
    path.setAttribute("d",
      `M ${this.cx - 3.5} ${this.cy + 10} L ${this.cx} ${this.cy - this.r + 8} ` +
      `L ${this.cx + 3.5} ${this.cy + 10} Z`);
    path.setAttribute("fill", "#e6edf5");
    this.needle.appendChild(path);
    el("circle", { cx: this.cx, cy: this.cy, r: 5.5, fill: "#22304a",
                   stroke: "#44597c" });
    if (def.icon) {
      el("text", { x: 100, y: 44, "text-anchor": "middle", "font-size": "13px" }).textContent = def.icon;
    }
    el("text", { x: 100, y: 58, "text-anchor": "middle",
                 class: "g-name" }).textContent = def.label.toUpperCase();
    this.readout = el("text", { x: 100, y: 140, "text-anchor": "middle",
                                class: "g-val" });
    el("text", { x: 100, y: 154, "text-anchor": "middle",
                 class: "g-unit" }).textContent = def.unit;
    const wrap = document.createElement("div");
    wrap.className = "gauge";
    wrap.appendChild(svg);
    mount.appendChild(wrap);
    this.redlineFrom = def.zones.length >= 2
      ? def.zones[def.zones.length - 2][0] : this.max;
    this.svgEl = svg;
  }
  valueToAngle(v) {
    const frac = clamp((v - this.min) / (this.max - this.min), 0, 1);
    return -135 + frac * 270;
  }
  polar(angDeg, radius) {
    const rad = (angDeg - 90) * Math.PI / 180;
    return [this.cx + radius * Math.cos(rad), this.cy + radius * Math.sin(rad)];
  }
  arcPath(fromV, toV) {
    const a1 = this.valueToAngle(fromV), a2 = this.valueToAngle(toV);
    const [sx, sy] = this.polar(a1, this.r);
    const [ex, ey] = this.polar(a2, this.r);
    const large = (a2 - a1) > 180 ? 1 : 0;
    return `M ${sx} ${sy} A ${this.r} ${this.r} 0 ${large} 1 ${ex} ${ey}`;
  }
  update(value) {
    const v = clamp(Number.isFinite(+value) ? +value : 0, this.min, this.max);
    this.needle.setAttribute("transform",
      `rotate(${this.valueToAngle(v)} ${this.cx} ${this.cy})`);
    this.readout.textContent =
      v.toFixed(this.def.decimals !== undefined ? this.def.decimals : 0);
    this.svgEl.classList.toggle("over", v >= this.redlineFrom);
  }
}

const GAUGE_DEFS = [
  { id: "rpm",        label: "Prop RPM",       unit: "rpm",  min: 0,    max: 3000, icon: "⏱",
    zones: [[2100, "#39d353"], [2600, "#e3b341"], [3000, "#f85149"]] },
  { id: "egt_degf",   label: "EGT",            unit: "\u00B0F", min: 800, max: 1900, icon: "🔥",
    zones: [[1200, "#39d353"], [1650, "#e3b341"], [1900, "#f85149"]], decimals: 0 },
  { id: "cht_degf",   label: "CHT",            unit: "\u00B0F", min: 100, max: 550, icon: "🌡️",
    zones: [[200, "#39d353"], [435, "#e3b341"], [550, "#f85149"]] },
  { id: "oil_temp_f", label: "Oil Temp",       unit: "\u00B0F", min: 60,  max: 250, icon: "🌡️",
    zones: [[120, "#e3b341"], [220, "#39d353"], [250, "#f85149"]] },
  { id: "oil_psi",    label: "Oil Press",      unit: "psi",  min: 0,    max: 115, icon: "🛢️",
    zones: [[25, "#f85149"], [95, "#39d353"], [115, "#f85149"]] },
  { id: "ff_gph",     label: "Fuel Flow",      unit: "gph",  min: 0,    max: 15, icon: "⛽",
    zones: [[10.5, "#39d353"], [13, "#e3b341"], [15, "#f85149"]], decimals: 1 },
  { id: "mp_inhg",    label: "Manifold Press", unit: "inHg", min: 10,   max: 32, icon: "🧭",
    zones: [[24, "#e3b341"], [29, "#39d353"], [32, "#f85149"]], decimals: 1 },
];
const gauges = {};
function buildGauges() {
  const mount = $("gauges");
  GAUGE_DEFS.forEach(d => { gauges[d.id] = new Gauge(d, mount); });
  Object.values(gauges).forEach(g => g.update(0));
}

/* =========================================================================
   3D SCENE  (Three.js — loads Cessna172 from cessna3d.js, optional GLTF)
   ========================================================================= */
const Scene3D = (() => {
  let active = false;
  let renderer, scene, camera, clock;
  let cessna = null;                  // Cessna172 instance

  function initThree(container) {
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b1220);
    scene.fog = new THREE.Fog(0x0b1220, 30, 90);
    camera = new THREE.PerspectiveCamera(
      48, container.clientWidth / container.clientHeight, 0.1, 300);
    camera.position.set(11, 5.5, 14);
    camera.lookAt(0, 0.8, 0);

    // lights
    scene.add(new THREE.HemisphereLight(0x8899bb, 0x1a2436, 0.85));
    const sun = new THREE.DirectionalLight(0xffffff, 0.9);
    sun.position.set(7, 12, 5);
    scene.add(sun);
    const fill = new THREE.DirectionalLight(0x6688aa, 0.3);
    fill.position.set(-5, 3, -4);
    scene.add(fill);

    // ground grid
    const grid = new THREE.GridHelper(90, 45, 0x24304a, 0x141d30);
    grid.position.y = -0.01;
    scene.add(grid);

    // Build the detailed Cessna 172P
    cessna = new Cessna172(scene);

    window.addEventListener("resize", () => {
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    });

    if (CFG.MODEL_URL) tryLoadGLTF(CFG.MODEL_URL);

    clock = new THREE.Clock();
    renderer.setAnimationLoop(tick);
    $("render-mode").textContent = "Three.js (Cessna 172P)";
  }

  /* ---- optional GLTF model -------------------------------------------- */
  function tryLoadGLTF(url) {
    if (!THREE.GLTFLoader) return;
    new THREE.GLTFLoader().load(url, (gltf) => {
      const keep = [];
      scene.traverse(o => { if (o.isLight || o.isGridHelper) keep.push(o); });
      while (scene.children.length) scene.remove(scene.children[0]);
      keep.forEach(o => scene.add(o));
      gltf.scene.traverse(o => { if (o.isMesh) o.castShadow = true; });
      scene.add(gltf.scene);
      const box = new THREE.Box3().setFromObject(gltf.scene);
      const size = box.getSize(new THREE.Vector3());
      const s = 9 / Math.max(size.x, size.y, size.z, 0.001);
      gltf.scene.scale.multiplyScalar(s);
      gltf.scene.position.y -= box.min.y * s;
      cessna = null;          // GLTF replaces procedural model
      $("render-mode").textContent = "Three.js (GLTF)";
    }, undefined, () => {});
  }

  /* ---- per-frame update ------------------------------------------------ */
  function tick() {
    const st = window.__twinState || {};

    if (cessna) {
      cessna.update(st);
    }

    // Camera shake based on anomaly score
    if (camera) {
      const baseScore = st.anomaly_score || 0;
      const thr = st.threshold || 0.045;
      if (baseScore > thr * 0.8) {
        // Start shaking if approaching or exceeding threshold
        const intensity = Math.min((baseScore - thr * 0.8) * 5, 0.5); // cap at 0.5
        camera.position.x = 11 + (Math.random() - 0.5) * intensity;
        camera.position.y = 5.5 + (Math.random() - 0.5) * intensity;
      } else {
        camera.position.set(11, 5.5, 14); // Reset to default
      }
    }

    renderer.render(scene, camera);
  }

  /* ---- public API ------------------------------------------------------ */
  function init(container) {
    try {
      if (!window.THREE) throw new Error("Three.js not loaded");
      initThree(container);
      active = true;
      return "three";
    } catch (err) {
      console.warn("Falling back to 2D:", err);
      enableFallback2D();
      return "2d";
    }
  }

  function enableFallback2D() {
    $("scene").style.display = "none";
    $("fallback2d").style.display = "flex";
    $("render-mode").textContent = "2D fallback";
    active = true;
  }

  function update2DFallback(st) {
    if (!active || cessna) return;
    const rpm = st.sensors ? (st.sensors.rpm || 0) : 0;
    const egt = st.sensors ? (st.sensors.egt_degf || 0) : 0;
    const anom = !!st.is_anomaly;
    const prop = $("fb-prop"), eng = $("fb-engine");
    if (!prop || !eng) return;
    const deg = ((performance.now() * 0.001) * (rpm / 60) * 360 * 0.08) % 360;
    prop.setAttribute("transform", `rotate(${deg.toFixed(1)} 160 196)`);
    const egtn = clamp((egt - 400) / 1400, 0, 1);
    const r = Math.round(anom ? 255 : 30 + egtn * 225);
    const g = Math.round(anom ? 42 : 102 - egtn * 70);
    const b = Math.round(anom ? 26 : 255 - egtn * 180);
    eng.setAttribute("fill", `rgb(${r},${g},${b})`);
    eng.classList.toggle("blink-red", anom);
  }

  return { init, is3D: () => !!renderer, updateFallback: update2DFallback };
})();

/* =========================================================================
   FAULT INJECTION CONTROLS
   ========================================================================= */
let wsRef = null;        // set on WS connect
let currentFault = null;

function sendCommand(cmd) {
  if (wsRef && wsRef.readyState === WebSocket.OPEN) {
    wsRef.send(JSON.stringify(cmd));
  }
}
const sendWsCmd = sendCommand;

function initFaultControls() {
  const slider = $("severity-slider");
  const sevVal = $("severity-val");
  if (slider) {
    slider.addEventListener("input", () => {
      sevVal.textContent = slider.value + "%";
      if (currentFault) {
        sendCommand({ cmd: "severity", type: currentFault, severity: parseInt(slider.value) });
      }
    });
  }
  document.querySelectorAll(".fault-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const fault = btn.dataset.fault;
      const sev = parseInt((slider || {}).value || 60);
      if (fault === "normal" || fault === "reset") {
        sendCommand({ cmd: "clear" });
        currentFault = null;
        document.querySelectorAll(".fault-btn").forEach(b => b.classList.remove("active"));
        logEvent("Engine reset - all faults cleared", "good");
      } else {
        sendCommand({ cmd: "fault", type: fault, severity: sev });
        currentFault = fault;
        document.querySelectorAll(".fault-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        logEvent("Fault injected: " + fault.toUpperCase() + " (severity " + sev + "%)", "warn");
      }
    });
  });

  const btnMitigate = $("btn-mitigate");
  if (btnMitigate) {
    btnMitigate.addEventListener("click", () => {
      sendCommand({ cmd: "mitigate" });
      logEvent("Auto-Pilot action executed - mitigating...", "good");
      // Optionally hide the panel temporarily until server confirms
      $("mitigation-panel").style.display = "none";
    });
  }
}

/* =========================================================================
   HISTORICAL CHARTS (Chart.js)
   ========================================================================= */
const SensorHistory = (() => {
  const MAX = CFG.CHART_HISTORY;
  const labels = [];
  const data = {};
  GAUGE_DEFS.forEach(d => { data[d.id] = []; });

  let chartSensors = null, chartAI = null, chartLandscape = null;
  const landscapeData = []; // Store past few points to show a trail

  function init() {
    if (!window.Chart) { console.warn("Chart.js not loaded"); return; }
    const sCtx = $("chart-sensors");
    const aCtx = $("chart-ai");
    const lCtx = $("chart-landscape");
    if (!sCtx || !aCtx) return;

    const sensorColors = {
      rpm: "#58a6ff", egt_degf: "#f85149", cht_degf: "#e3b341",
      oil_temp_f: "#f0883e", oil_psi: "#39d353", ff_gph: "#a371f7",
      mp_inhg: "#79c0ff"
    };
    const sensorDatasets = GAUGE_DEFS.map(d => ({
      label: d.label, data: [], borderColor: sensorColors[d.id] || "#888",
      borderWidth: 1.2, pointRadius: 0, tension: 0.3, yAxisID: "y",
    }));

    chartSensors = new Chart(sCtx, {
      type: "line",
      data: { labels: [], datasets: sensorDatasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        animation: false,
        plugins: { legend: { display: true, labels: { color: "#8aa0b4", font: { size: 10 } } } },
        scales: {
          x: { display: true, ticks: { color: "#556677", maxTicksLimit: 8, font: { size: 9 } },
               grid: { color: "#1b2740" } },
          y: { display: true, ticks: { color: "#556677", font: { size: 9 } },
               grid: { color: "#1b2740" } },
        },
      },
    });

    chartAI = new Chart(aCtx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          { label: "Anomaly Score", data: [], borderColor: "#f85149",
            borderWidth: 1.5, pointRadius: 0, tension: 0.3, yAxisID: "y" },
          { label: "Health Index", data: [], borderColor: "#39d353",
            borderWidth: 1.5, pointRadius: 0, tension: 0.3, yAxisID: "y1" },
          { label: "RUL (h)", data: [], borderColor: "#58a6ff",
            borderWidth: 1.5, pointRadius: 0, tension: 0.3, yAxisID: "y1" },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        animation: false,
        plugins: { legend: { display: true, labels: { color: "#8aa0b4", font: { size: 10 } } } },
        scales: {
          x: { display: true, ticks: { color: "#556677", maxTicksLimit: 8, font: { size: 9 } },
               grid: { color: "#1b2740" } },
          y: { type: "linear", position: "left", title: { display: true, text: "Anomaly", color: "#f85149", font: { size: 9 } },
               ticks: { color: "#556677", font: { size: 9 } }, grid: { color: "#1b2740" } },
          y1: { type: "linear", position: "right",
                title: { display: true, text: "Health / RUL", color: "#39d353", font: { size: 9 } },
                ticks: { color: "#556677", font: { size: 9 } }, grid: { drawOnChartArea: false } },
        },
      },
    });

    if (lCtx) {
      chartLandscape = new Chart(lCtx, {
        type: "scatter",
        data: {
          datasets: [{
            label: "Health State",
            data: landscapeData,
            backgroundColor: (ctx) => {
              const idx = ctx.dataIndex;
              const max = ctx.dataset.data.length;
              // Fade out older points
              const alpha = Math.max(0.1, idx / max);
              return `rgba(88, 166, 255, ${alpha})`;
            },
            pointRadius: (ctx) => {
              return ctx.dataIndex === ctx.dataset.data.length - 1 ? 6 : 3;
            }
          }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          animation: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { min: -2, max: 2, grid: { color: "#1b2740" }, title: { display: true, text: "Component 1 (Stress)", color: "#556677" }, ticks: { color: "#556677"} },
            y: { min: -2, max: 2, grid: { color: "#1b2740" }, title: { display: true, text: "Component 2 (Thermal)", color: "#556677" }, ticks: { color: "#556677"} },
          }
        }
      });
    }
  }

  function push(st) {
    if (!chartSensors || !chartAI) return;
    const ts = fmtClock(st.timestamp);
    labels.push(ts);
    if (labels.length > MAX) labels.shift();
    GAUGE_DEFS.forEach(d => {
      const val = st.sensors ? (st.sensors[d.id] || 0) : 0;
      data[d.id].push(val);
      if (data[d.id].length > MAX) data[d.id].shift();
    });
    chartSensors.data.labels = [...labels];
    GAUGE_DEFS.forEach((d, i) => {
      chartSensors.data.datasets[i].data = [...data[d.id]];
    });
    chartSensors.update("none");

    chartAI.data.labels = [...labels];
    chartAI.data.datasets[0].data.push(st.anomaly_score || 0);
    chartAI.data.datasets[1].data.push((st.health || 1) * 100);
    chartAI.data.datasets[2].data.push(st.rul_h || 0);
    ["", "", ""].forEach((_, i) => {
      if (chartAI.data.datasets[i].data.length > MAX)
        chartAI.data.datasets[i].data.shift();
    });
    chartAI.update("none");

    if (chartLandscape && st.normalized && st.normalized.length >= 7) {
      // Pseudo-PCA projection of the 7 normalized sensor values
      // [rpm, egt, cht, oil_temp, oil_psi, ff, mp]
      const n = st.normalized;
      const pc1 = (n[0]*0.4 + n[1]*0.4 + n[2]*0.4 + n[3]*0.4 - n[4]*0.4 + n[5]*0.4 - n[6]*0.4) - 0.5;
      const pc2 = (-n[0]*0.5 + n[1]*0.5 + n[2]*0.2 + n[3]*0.5 + n[4]*0.0 - n[5]*0.5 + n[6]*0.0);
      
      landscapeData.push({ x: pc1 * 3, y: pc2 * 3 }); // Scale up for visibility
      if (landscapeData.length > 50) landscapeData.shift(); // Keep trail of 50 points
      
      // Update color if anomaly
      chartLandscape.data.datasets[0].backgroundColor = (ctx) => {
        const idx = ctx.dataIndex;
        const max = ctx.dataset.data.length;
        const alpha = Math.max(0.1, idx / max);
        return st.is_anomaly ? `rgba(248, 81, 73, ${alpha})` : `rgba(88, 166, 255, ${alpha})`;
      };
      
      chartLandscape.update("none");
    }
  }

  return { init, push };
})();

/* =========================================================================
   EXPLAINABILITY PANEL
   ========================================================================= */
function updateExplainPanel(explain, isAnomaly) {
  const panel = $("explain-panel");
  if (!panel || !explain) return;
  if (!isAnomaly) {
    panel.innerHTML = '<div style="color:var(--dim);font-size:11px">Engine nominal &mdash; no deviations</div>';
    return;
  }
  const entries = Object.entries(explain)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);
  const maxDev = Math.max(entries[0][1], 0.01);
  const SENSOR_LABELS = {
    rpm: "RPM", egt_degf: "EGT", cht_degf: "CHT",
    oil_temp_f: "Oil Temp", oil_psi: "Oil Press",
    ff_gph: "Fuel Flow", mp_inhg: "Manifold P"
  };
  const barColor = (dev) => {
    const ratio = dev / Math.max(maxDev, 0.001);
    if (ratio > 0.7) return "var(--red)";
    if (ratio > 0.3) return "var(--amber)";
    return "var(--green)";
  };
  panel.innerHTML = entries.map(([k, v]) => {
    const pct = clamp((v / maxDev) * 100, 0, 100);
    return `<div class="explain-bar">
      <span class="eb-label">${SENSOR_LABELS[k] || k}</span>
      <div class="eb-track"><div class="eb-fill" style="width:${pct}%;background:${barColor(v)}"></div></div>
      <span class="eb-val" style="color:${barColor(v)}">${v.toFixed(3)}</span>
    </div>`;
  }).join("");
}

/* =========================================================================
   ML PREDICTION PANEL  (fed by backend's call_ml_predict → predict_server)
   ========================================================================= */
function updateMLPanel(st) {
  const badge = $("ml-badge");
  const rul   = $("ml-rul");
  const conf  = $("ml-conf");
  const mode  = $("ml-mode");
  const win   = $("ml-window");
  const grid  = $("prob-grid");

  if (!badge) return;                       // panel not in DOM

  const fault = st.ml_fault;
  if (!fault) {
    // ML server unreachable – show waiting state
    badge.className = "ml-badge ml-none";
    badge.textContent = "WAITING…";
    conf.textContent   = "–";
    mode.textContent   = "connecting to ML API…";
    return;
  }

  // Determine severity class
  const isNone = fault === "none";
  const isCrit = ["multi"].includes(fault);
  if (isNone) {
    badge.className = "ml-badge ml-none";
    badge.textContent = "NORMAL";
  } else if (isCrit) {
    badge.className = "ml-badge ml-crit";
    badge.textContent = fault.toUpperCase();
  } else {
    badge.className = "ml-badge ml-fault";
    badge.textContent = fault.toUpperCase();
  }

  // Confidence
  const confVal = st.ml_confidence ?? 0;
  conf.textContent = (confVal * 100).toFixed(1) + "%";

  // Mode
  mode.textContent = st.mode || "–";

  // Window status
  if (st.ml_window_filled) {
    win.textContent = "ready (" + (st.ml_rul_h != null ? "predicting" : "buffering") + ")";
  } else {
    win.textContent = "filling…";
  }

  // RUL
  if (st.ml_rul_h != null) {
    const hours = st.ml_rul_h;
    rul.textContent = hours.toFixed(1) + " h";
    // Update RUL bar (scale: 0 = empty, 1.0 = full)
    const bar = $("ml-rul-bar");
    if (bar) {
      bar.style.width = clamp(hours * 100, 0, 100).toFixed(1) + "%";
    }
  } else {
    rul.textContent = "– h";
  }

  // Probability grid
  if (grid && st.ml_probabilities) {
    const entries = Object.entries(st.ml_probabilities)
      .sort((a, b) => b[1] - a[1]);
    grid.innerHTML = entries.map(([name, prob]) => {
      const pct = (prob * 100).toFixed(1);
      return `<div class="prob-row">
        <span class="label">${name}</span>
        <div class="track"><div class="fill" style="width:${pct}%"></div></div>
        <span class="val">${pct}%</span>
      </div>`;
    }).join("");
  }

  // Mitigation Auto-Pilot
  const mitPanel = $("mitigation-panel");
  const mitText = $("mitigation-text");
  if (mitPanel && mitText) {
    if (st.is_anomaly || (st.ml_rul_h != null && st.ml_rul_h < 15)) {
      mitPanel.style.display = "block";
      // Determine recommended action based on fault
      if (fault === "oil" || fault === "oiltemp") {
        mitText.textContent = "AI recommends: LAND IMMEDIATELY (Engine Damage Imminent)";
      } else if (fault === "egt" || fault === "cht") {
        mitText.textContent = "AI recommends: ENRICH MIXTURE & REDUCE THROTTLE";
      } else {
        mitText.textContent = "AI recommends: THROTTLE DOWN TO 2100 RPM";
      }
    } else {
      mitPanel.style.display = "none";
    }
  }
}

/* =========================================================================
   ACTIVE FAULTS DISPLAY
   ========================================================================= */
function updateActiveFaults(faults) {
  const el = $("active-faults");
  if (!el) return;
  if (!faults || faults.length === 0) {
    el.innerHTML = '<div style="color:var(--dim);font-size:11px">No active faults</div>';
    return;
  }
  el.innerHTML = faults.map(f => {
    const name = typeof f === "string" ? f : f.name;
    const sev = typeof f === "object" ? f.severity : 100;
    const dur = typeof f === "object" ? f.duration_s : null;
    return `<div class="af-item">
      <span>${name.toUpperCase()}</span>
      <span>${sev}%${dur !== null ? " (" + dur.toFixed(0) + "s)" : ""}</span>
    </div>`;
  }).join("");
}

/* =========================================================================
   STATE APPLICATION  (WebSocket JSON -> DOM)
   ========================================================================= */
let prevState = { is_anomaly: false, status: "", mode: null };

function applyState(st) {
  window.__twinState = st;

  GAUGE_DEFS.forEach(d => {
    if (gauges[d.id] && st.sensors) gauges[d.id].update(st.sensors[d.id]);
  });

  const badge = $("status-badge");
  if (st.status && st.status !== prevState.status) {
    badge.className = "badge st-" + st.status;
    badge.textContent = st.status;
    if (prevState.status !== "") {
      if (st.status === "WARNING")
        logEvent("Status degraded: WARNING (score " + st.anomaly_score + " > thr " + st.threshold + ")", "warn");
      if (st.status === "CRITICAL")
        logEvent("CRITICAL ENGINE CONDITION - score=" + st.anomaly_score, "crit");
      if (st.status === "NOMINAL")
        logEvent("Parameters back within nominal envelope", "good");
    }
    prevState.status = st.status;
  }

  if (!!st.is_anomaly !== !!prevState.is_anomaly) {
    const bannerEl = $("banner");
    if (bannerEl) bannerEl.style.display = st.is_anomaly ? "block" : "none";
    if (st.is_anomaly) {
      logEvent("ANOMALY DETECTED (score " + st.anomaly_score + " > threshold " + st.threshold + ")", "crit");
    } else {
      logEvent("Anomaly cleared - readings normalised", "good");
    }
    prevState.is_anomaly = st.is_anomaly;
  }

  $("rul-value").textContent =
    (st.rul_h != null) ? st.rul_h.toFixed(1) : "--";
  $("health-val").textContent =
    st.health != null ? (st.health * 100).toFixed(1) + "%" : "--";
  $("health-fill").style.width =
    st.health != null ? (st.health * 100).toFixed(1) + "%" : "100%";
  $("score-val").textContent =
    st.anomaly_score != null ? st.anomaly_score.toFixed(4) : "--";
  $("thr-val").textContent =
    st.threshold != null ? st.threshold.toFixed(4) : "--";
  $("mode-val").textContent = (st.mode || "--") +
    (st.engine_running ? "" : "  (engine off)");

  const faults = st.faults || [];
  $("faults-val").textContent =
    faults.length ? faults.map(f => typeof f === "string" ? f : f.name).join(", ") : "none";

  const maxDur = faults.reduce((m, f) => {
    const d = (typeof f === "object" && f.duration_s != null) ? f.duration_s : 0;
    return d > m ? d : m;
  }, 0);
  $("duration-val").textContent = maxDur > 0 ? maxDur.toFixed(1) + " s" : "--";

  $("rul-split").textContent =
    (st.rul_lstm_h != null ? st.rul_lstm_h : "-") + " / " +
    (st.rul_synth_h != null ? st.rul_synth_h : "-");
  $("rul-detail").textContent = st.rul_lstm_h != null
    ? "Remaining Useful Life (LSTM + synthetic)"
    : "Remaining Useful Life (synthetic degradation model)";

  updateActiveFaults(faults);
  updateExplainPanel(st.explain, st.is_anomaly);
  updateMLPanel(st);
  updateMLPlayground(st);
  SensorHistory.push(st);

  if (prevState.mode !== st.mode) {
    if (prevState.mode !== null) logEvent("Data mode: " + st.mode, "info");
    prevState.mode = st.mode;
  }
  Scene3D.updateFallback(st);
}

/* =========================================================================
   WEBSOCKET CONNECTION MANAGER
   ========================================================================= */
let ws = null, retries = 0;

function setConnState(ok, text) {
  $("ws-dot").className = "dot " + (ok ? "on" : "err");
  $("ws-text").textContent = text;
}

function connectWS() {
  setConnState(false, "connecting\u2026");
  try { ws = new WebSocket(CFG.WS_URL); }
  catch (err) { setConnState(false, "bad WS URL"); scheduleReconnect(); return; }

  ws.onopen = () => {
    retries = 0;
    wsRef = ws;
    setConnState(true, "connected \u00B7 :8765");
    logEvent("Telemetry link established", "good");
  };
  ws.onmessage = (ev) => {
    try { applyState(JSON.parse(ev.data)); }
    catch (err) { console.error("Bad JSON from backend:", err); }
  };
  ws.onclose = () => {
    wsRef = null;
    setConnState(false, "disconnected - retrying");
    scheduleReconnect();
  };
  ws.onerror = () => ws.close();
}

function scheduleReconnect() {
  if (ws && ws.readyState === WebSocket.OPEN) return;
  const delay = Math.min(CFG.RECONNECT_BASE_MS * Math.pow(2, retries++),
                         CFG.RECONNECT_MAX_MS);
  setTimeout(connectWS, delay);
}


/* =========================================================================
   ML PLAYGROUND & UNIFIED VIEW CONTROLLER
   ========================================================================= */
function updateMLPlayground(st) {
  const s = st.sensors || {};
  
  // 1. Update 7 Mini Digital Cards (Playground Tab)
  if ($("pg-v-rpm")) $("pg-v-rpm").textContent = Math.round(s.rpm ?? 0);
  if ($("pg-v-egt")) $("pg-v-egt").textContent = Math.round(s.egt_degf ?? 0);
  if ($("pg-v-cht")) $("pg-v-cht").textContent = Math.round(s.cht_degf ?? 0);
  if ($("pg-v-oil_temp")) $("pg-v-oil_temp").textContent = Math.round(s.oil_temp_f ?? 0);
  if ($("pg-v-oil_press")) $("pg-v-oil_press").textContent = (s.oil_psi ?? 0).toFixed(1);
  if ($("pg-v-fuel_flow")) $("pg-v-fuel_flow").textContent = (s.ff_gph ?? 0).toFixed(1);
  if ($("pg-v-manifold")) $("pg-v-manifold").textContent = (s.mp_inhg ?? 0).toFixed(1);

  const fault = st.ml_fault || "none";
  const confVal = st.ml_confidence ?? 0;
  const rulVal = st.ml_rul_h;
  const isHealthy = fault === "none";
  const isCrit = ["multi"].includes(fault) || (rulVal != null && rulVal < 0.1);

  // 2. Engine Mode & LED (Playground Tab)
  const led = $("pg-led-state");
  const modeText = $("pg-engine-mode");
  if (led && modeText) {
    if (isHealthy) {
      led.className = "led on";
      modeText.textContent = "NORMAL";
    } else if (isCrit) {
      led.className = "led crit";
      modeText.textContent = fault.toUpperCase() + " (CRITICAL)";
    } else {
      led.className = "led warn";
      modeText.textContent = fault.toUpperCase();
    }
  }

  // 3. AI Prediction Results (Playground Tab)
  const pgBadge = $("pg-fault-badge");
  if (pgBadge) {
    pgBadge.textContent = fault.toUpperCase();
    pgBadge.className = "fault-badge-large " +
      (isHealthy ? "fault-none" : isCrit ? "fault-critical" : "fault-active");
  }

  if ($("pg-confidence-line")) {
    $("pg-confidence-line").textContent = "confidence: " + (confVal * 100).toFixed(0) + "%";
  }

  if ($("pg-rul-out")) {
    $("pg-rul-out").textContent = rulVal != null ? rulVal.toFixed(2) + " h" : "buffering...";
  }

  if ($("pg-status-out")) {
    $("pg-status-out").textContent = rulVal != null
      ? (rulVal > 0.3 ? "HEALTHY" : rulVal > 0.1 ? "WARNING" : "CRITICAL")
      : "collecting window";
    $("pg-status-out").style.color = rulVal != null
      ? (rulVal > 0.3 ? "var(--green)" : rulVal > 0.1 ? "var(--amber)" : "var(--red)")
      : "var(--dim)";
  }

  if ($("pg-elapsed-out")) {
    $("pg-elapsed-out").textContent = (st.t_s ? Math.round(st.t_s) : "0") + " s";
  }

  if ($("pg-window-out")) {
    $("pg-window-out").textContent = st.ml_window_filled ? "15 / 15 (Ready)" : "filling...";
  }

  if ($("pg-rul-fill")) {
    const pct = rulVal != null ? clamp((rulVal / 0.5) * 100, 0, 100) : 0;
    $("pg-rul-fill").style.width = pct + "%";
  }

  // 4. Fault Probabilities Grid (Playground Tab)
  const pgGrid = $("pg-prob-grid");
  if (pgGrid && st.ml_probabilities) {
    const entries = Object.entries(st.ml_probabilities).sort((a,b) => b[1] - a[1]);
    pgGrid.innerHTML = entries.map(([name, p]) => {
      const pct = (p * 100).toFixed(0);
      return `<div class="prob-row">
        <span class="label">${name}</span>
        <div class="track"><div class="fill" style="width:${pct}%"></div></div>
        <span class="val">${pct}%</span>
      </div>`;
    }).join("");
  }

  // 5. Header ML Health Pill
  const mlPill = $("ml-health-pill");
  if (mlPill) {
    if (st.ml_fault) {
      mlPill.className = "status-pill pill-ok";
      mlPill.innerHTML = '<i class="fa-solid fa-brain"></i> ML: ' + fault.toUpperCase();
    } else {
      mlPill.className = "status-pill pill-err";
      mlPill.innerHTML = '<i class="fa-solid fa-brain"></i> ML: offline';
    }
  }

  // 6. Unified View Elements
  if ($("u-fault-badge")) {
    const ub = $("u-fault-badge");
    ub.textContent = fault.toUpperCase();
    ub.className = "fault-badge-large " +
      (isHealthy ? "fault-none" : isCrit ? "fault-critical" : "fault-active");
  }
  if ($("u-rul-out")) $("u-rul-out").textContent = rulVal != null ? rulVal.toFixed(2) + " h" : "--";
  if ($("u-conf-out")) $("u-conf-out").textContent = (confVal * 100).toFixed(0) + "%";
  if ($("u-health-out")) $("u-health-out").textContent = st.health != null ? (st.health * 100).toFixed(0) + "%" : "--";
  if ($("u-rul-fill")) {
    const upct = rulVal != null ? clamp((rulVal / 0.5) * 100, 0, 100) : 0;
    $("u-rul-fill").style.width = upct + "%";
  }
  const uGrid = $("u-prob-grid");
  if (uGrid && st.ml_probabilities) {
    const entries = Object.entries(st.ml_probabilities).sort((a,b) => b[1] - a[1]);
    uGrid.innerHTML = entries.slice(0, 5).map(([name, p]) => {
      const pct = (p * 100).toFixed(0);
      return `<div class="prob-row">
        <span class="label">${name}</span>
        <div class="track"><div class="fill" style="width:${pct}%"></div></div>
        <span class="val">${pct}%</span>
      </div>`;
    }).join("");
  }
}

/* =========================================================================
   TAB NAVIGATION SYSTEM
   ========================================================================= */
function initTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");
      tabs.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      document.querySelectorAll(".view-container").forEach(c => c.classList.remove("active"));
      const targetEl = $(targetId);
      if (targetEl) targetEl.classList.add("active");

      // Trigger 3D WebGL resize
      setTimeout(() => {
        if (window.dispatchEvent) window.dispatchEvent(new Event("resize"));
      }, 60);
    });
  });
}

/* =========================================================================
   PLAYGROUND & UNIFIED CONTROLS INITIALIZATION
   ========================================================================= */
function initPlaygroundControls() {
  // Fault Selector (Playground)
  const pgFault = $("pg-fault-select");
  if (pgFault) {
    pgFault.addEventListener("change", (e) => {
      const val = e.target.value;
      const sev = +($("pg-severity") ? $("pg-severity").value : 60);
      if (val === "none") {
        sendWsCmd({ cmd: "clear" });
      } else {
        sendWsCmd({ cmd: "fault", type: val, severity: sev });
      }
    });
  }

  // Severity Slider (Playground)
  const pgSev = $("pg-severity");
  if (pgSev) {
    pgSev.addEventListener("input", (e) => {
      const val = e.target.value;
      if ($("pg-sev-val")) $("pg-sev-val").textContent = val + "%";
      const currentFault = pgFault ? pgFault.value : "none";
      if (currentFault !== "none") {
        sendWsCmd({ cmd: "severity", type: currentFault, severity: +val });
      }
    });
  }

  // Custom EGT Override (Playground)
  const pgEgt = $("pg-egt-ovr");
  if (pgEgt) {
    pgEgt.addEventListener("input", (e) => {
      const val = +e.target.value;
      if ($("pg-egt-ovr-val")) $("pg-egt-ovr-val").textContent = val;
      if (val === 1290) {
        sendWsCmd({ cmd: "override", sensor: "egt_degf", value: null });
      } else {
        sendWsCmd({ cmd: "override", sensor: "egt_degf", value: val });
      }
    });
  }

  // Reset Button (Playground)
  const pgReset = $("pg-btn-reset");
  if (pgReset) {
    pgReset.addEventListener("click", () => {
      if (pgEgt) { pgEgt.value = 1290; if ($("pg-egt-ovr-val")) $("pg-egt-ovr-val").textContent = "1290"; }
      if (pgFault) pgFault.value = "none";
      if (pgSev) { pgSev.value = 60; if ($("pg-sev-val")) $("pg-sev-val").textContent = "60%"; }
      sendWsCmd({ cmd: "clear" });
      sendWsCmd({ cmd: "override", sensor: "egt_degf", value: null });
      try { fetch("http://localhost:5000/reset", { method: "POST" }); } catch(e){}
    });
  }

  // Unified Controls Wiring
  const uFault = $("u-fault-select");
  if (uFault) {
    uFault.addEventListener("change", (e) => {
      const val = e.target.value;
      const sev = +($("u-severity") ? $("u-severity").value : 60);
      if (val === "none") sendWsCmd({ cmd: "clear" });
      else sendWsCmd({ cmd: "fault", type: val, severity: sev });
    });
  }

  const uSev = $("u-severity");
  if (uSev) {
    uSev.addEventListener("input", (e) => {
      const val = e.target.value;
      if ($("u-sev-val")) $("u-sev-val").textContent = val + "%";
      const cur = uFault ? uFault.value : "none";
      if (cur !== "none") sendWsCmd({ cmd: "severity", type: cur, severity: +val });
    });
  }

  const uEgt = $("u-egt-ovr");
  if (uEgt) {
    uEgt.addEventListener("input", (e) => {
      const val = +e.target.value;
      if ($("u-egt-val")) $("u-egt-val").textContent = val + " °F";
      if (val === 1290) sendWsCmd({ cmd: "override", sensor: "egt_degf", value: null });
      else sendWsCmd({ cmd: "override", sensor: "egt_degf", value: val });
    });
  }

  const uReset = $("u-btn-reset");
  if (uReset) {
    uReset.addEventListener("click", () => {
      if (uEgt) { uEgt.value = 1290; if ($("u-egt-val")) $("u-egt-val").textContent = "1290 °F"; }
      if (uFault) uFault.value = "none";
      if (uSev) { uSev.value = 60; if ($("u-sev-val")) $("u-sev-val").textContent = "60%"; }
      sendWsCmd({ cmd: "clear" });
      sendWsCmd({ cmd: "override", sensor: "egt_degf", value: null });
      try { fetch("http://localhost:5000/reset", { method: "POST" }); } catch(e){}
    });
  }

  const uHeal = $("u-btn-heal");
  if (uHeal) {
    uHeal.addEventListener("click", () => {
      sendWsCmd({ cmd: "heal" });
    });
  }
}

/* =========================================================================
   BOOT
   ========================================================================= */
window.addEventListener("DOMContentLoaded", () => {
  buildGauges();
  const mode = Scene3D.init($("scene"));
  logEvent(mode === "three" ? "3D twin ready" : "WebGL unavailable - 2D fallback", mode === "three" ? "good" : "warn");
  initFaultControls();
  initTabs();
  initPlaygroundControls();
  SensorHistory.init();
  connectWS();
});
