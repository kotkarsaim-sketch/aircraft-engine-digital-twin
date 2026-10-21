/* C172P ML Prediction Playground frontend.
   Manually controls Cessna 172P simulator + calls Flask /predict API. */
"use strict";

const $ = (id) => document.getElementById(id);

const sensorKeys = ["rpm", "egt", "cht", "oil_temp", "oil_press",
                    "fuel_flow", "manifold"];

// Simulator state (mirrors sim_engine.py)
let state = {
  t: 0,
  fault: "none",
  severity: 0.6,
  sensors: { rpm: 2320, egt: 1290, cht: 372, oil_temp: 183,
             oil_press: 61, fuel_flow: 8.8, manifold: 27.3 },
};

const FAULT_LIB = {
  egt:     { egt: (s) => 1290 + s*560, oil_temp: (s) => 183 + s*20 },
  cht:     { cht: (s) => 372 + s*158,  oil_temp: (s) => 183 + s*18 },
  oil:     { oil_press: (s) => 61 - s*51, oil_temp: (s) => 183 + s*25 },
  oiltemp: { oil_temp: (s) => 183 + s*67 },
  ff:      { fuel_flow: (s) => 8.8 - s*6.8 },
  mp:      { manifold:  (s) => 27.3 - s*15.3 },
  multi:   { egt: (s)=>1290+s*560, oil_press:(s)=>61-s*51, oil_temp:(s)=>183+s*55 },
};

function stepSim(dt) {
  state.t += dt;
  const g = gauss;
  const t = state.t;
  // healthy baseline with wander + noise
  const s = {
    rpm: 2320 + 140*Math.sin(0.05*t) + g()*25,
    egt: 1290 + 30*Math.sin(0.033*t) + g()*9,
    cht: 372 + 14*Math.sin(0.021*t) + g()*5,
    oil_temp: 183 + 4*Math.sin(0.008*t) + g()*2,
    oil_press: 61 + 1.5*Math.sin(0.04*t) + g()*1.2,
    manifold: 27.3 + 0.4*Math.sin(0.05*t) + g()*0.1,
  };
  s.fuel_flow = 8.8 + (s.rpm-2320)/140*0.5 + g()*0.15;
  // apply fault
  if (state.fault !== "none" && FAULT_LIB[state.fault]) {
    for (const [k, fn] of Object.entries(FAULT_LIB[state.fault])) {
      s[k] = fn(state.severity);
    }
  }
  // manual EGT override
  if (manualEgt.active) s.egt = manualEgt.value;
  state.sensors = s;
}

function gauss() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
}

/* ---- UI wiring ---- */
let autoPredict = false;
const manualEgt = { active: false, value: 1290 };

function updateGauges() {
  const s = state.sensors;
  $("v-rpm").textContent = Math.round(s.rpm);
  $("v-egt").textContent = Math.round(s.egt);
  $("v-cht").textContent = Math.round(s.cht);
  $("v-oil_temp").textContent = Math.round(s.oil_temp);
  $("v-oil_press").textContent = s.oil_press.toFixed(1);
  $("v-fuel_flow").textContent = s.fuel_flow.toFixed(1);
  $("v-manifold").textContent = s.manifold.toFixed(1);
}

async function predict() {
  try {
    const resp = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sensors: state.sensors, t: state.t }),
    });
    const data = await resp.json();
    if (data.error) { console.error(data.error); return; }
    renderResult(data);
  } catch (err) {
    console.error("predict failed:", err);
  }
}

function renderResult(data) {
  const badge = $("fault-badge");
  badge.textContent = data.fault.toUpperCase();
  badge.className = "fault-badge " +
    (data.fault === "none" ? "fault-none" :
     data.rul_h != null && data.rul_h < 0.1 ? "fault-critical" : "fault-active");
  $("confidence-line").textContent =
    "confidence: " + (data.confidence*100).toFixed(0) + "%";
  $("rul-out").textContent =
    data.rul_h != null ? data.rul_h.toFixed(2) + " h" : "buffering...";
  $("status-out").textContent =
    data.rul_h != null
      ? (data.rul_h > 0.3 ? "HEALTHY" : data.rul_h > 0.1 ? "WARNING" : "CRITICAL")
      : "collecting window";
  $("elapsed-out").textContent = state.t.toFixed(0) + " s";
  $("window-out").textContent = data.window_filled ? "15 / 15" : "filling...";

  // RUL bar
  const rulPct = data.rul_h != null ? Math.min(100, (data.rul_h / 0.5) * 100) : 0;
  $("rul-fill").style.width = rulPct + "%";

  const grid = $("prob-grid");
  grid.innerHTML = "";
  Object.entries(data.probabilities || {})
    .sort((a,b) => b[1]-a[1])
    .forEach(([name, p]) => {
      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML =
        `<span class="p-label">${name}</span>
         <div class="p-track"><div class="p-fill" style="width:${(p*100).toFixed(0)}%"></div></div>
         <span class="p-val">${(p*100).toFixed(0)}%</span>`;
      grid.appendChild(row);
    });

  // Engine mode LED
  const led = $("led-state");
  const modeText = $("engine-mode");
  if (data.fault === "none") {
    led.className = "led on";
    modeText.textContent = "NORMAL";
  } else if (data.rul_h != null && data.rul_h < 0.1) {
    led.className = "led crit";
    modeText.textContent = data.fault.toUpperCase() + " (CRITICAL)";
  } else {
    led.className = "led warn";
    modeText.textContent = data.fault.toUpperCase();
  }
}

async function checkHealth() {
  try {
    const r = await fetch("/health");
    const d = await r.json();
    const pill = $("#health-pill");
    pill.textContent = d.status.toUpperCase();
    pill.className = "status-pill " + (d.status === "ok" ? "pill-ok" : "pill-err");
  } catch { /* server not up */ }
}

/* ---- events ---- */
$("fault-select").addEventListener("change", (e) => {
  state.fault = e.target.value;
});
$("severity").addEventListener("input", (e) => {
  state.severity = +e.target.value / 100;
  $("sev-val").textContent = e.target.value + "%";
});
$("egt-ovr").addEventListener("input", (e) => {
  const v = +e.target.value;
  manualEgt.value = v;
  manualEgt.active = (v !== 1290);
  $("egt-ovr-val").textContent = v;
});
$("btn-reset").addEventListener("click", () => {
  state.t = 0;
  state.fault = "none";
  state.severity = 0.6;
  manualEgt.active = false;
  $("egt-ovr").value = 1290;
  $("egt-ovr-val").textContent = "1290";
  $("fault-select").value = "none";
  $("severity").value = 60;
  $("sev-val").textContent = "60%";
  fetch("/reset", { method: "POST" });
  predict();
});
$("btn-auto").addEventListener("click", () => {
  autoPredict = !autoPredict;
  $("btn-auto").textContent =
    autoPredict ? "STOP PREDICT" : "AUTO-PREDICT (continuous)";
  if (autoPredict) {
    const tick = () => {
      if (!autoPredict) return;
      predict();
      setTimeout(tick, 500);
    };
    tick();
  }
});

/* ---- main loop (simulation) ---- */
let last = performance.now();
function loop(now) {
  const speed = +$("sim-speed").value;
  const dt = Math.min((now - last) / 1000, 0.1) * speed;
  last = now;
  stepSim(dt);
  updateGauges();
  requestAnimationFrame(loop);
}

window.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  predict();
  requestAnimationFrame(loop);
});