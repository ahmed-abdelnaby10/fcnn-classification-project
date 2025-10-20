// --- Config ---
const API_BASE = location.origin.replace(/:\d+$/, ":8000"); // adjust if backend port differs
const COLS = [
  "Pregnancies",
  "Glucose",
  "BloodPressure",
  "SkinThickness",
  "Insulin",
  "BMI",
  "DiabetesPedigreeFunction",
  "Age"
];

// --- Helpers ---
const $ = (s) => document.querySelector(s);
const el = {
  singleForm: $("#singleForm"),
  predictSingle: $("#predictSingle"),
  addToBatch: $("#addToBatch"),
  singleStatus: $("#singleStatus"),
  singleResult: $("#singleResult"),

  predictBatch: $("#predictBatch"),
  clearBatch: $("#clearBatch"),
  batchStatus: $("#batchStatus"),
  batchTable: $("#batchTable"),
  batchHeader: $("#batchHeader"),
  batchBody: $("#batchBody"),
  csvInput: $("#csvInput"),
  downloadCsv: $("#downloadCsv"),
};

function readSingleRaw() {
  const obj = {};
  for (const k of COLS) {
    const v = Number.parseFloat($("#" + k).value);
    if (Number.isNaN(v)) return { ok: false, err: `Missing/invalid value for ${k}` };
    obj[k] = v;
  }
  return { ok: true, raw: obj };
}

function rawToArray(raw) {
  // Ensure order
  return COLS.map((c) => Number(raw[c]));
}

function appendBatchRow(raw) {
  const tr = document.createElement("tr");
  for (const c of COLS) {
    const td = document.createElement("td");
    td.textContent = raw[c];
    tr.appendChild(td);
  }
  el.batchBody.appendChild(tr);
}

function tableToBatchArrays() {
  const rows = [];
  for (const tr of el.batchBody.querySelectorAll("tr")) {
    const tds = [...tr.querySelectorAll("td")].map((td) => Number(td.textContent));
    rows.push(tds);
  }
  return rows;
}

function resetSingleResult() {
  el.singleResult.textContent = "—";
  el.singleStatus.textContent = "";
}

// --- Init table header ---
(function initHeader() {
  el.batchHeader.innerHTML = "";
  for (const c of COLS) {
    const th = document.createElement("th");
    th.textContent = c;
    el.batchHeader.appendChild(th);
  }
})();

// --- Single Predict ---
el.predictSingle.addEventListener("click", async () => {
  resetSingleResult();
  const r = readSingleRaw();
  if (!r.ok) {
    el.singleStatus.textContent = r.err;
    return;
  }
  el.singleStatus.textContent = "…جاري التنبؤ";
  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // Safer raw dict; backend will reorder & scale
      body: JSON.stringify({ raw: r.raw }),
    });
    const j = await res.json();
    if (!res.ok) throw new Error(j.detail || "API error");
    el.singleResult.textContent = JSON.stringify(j, null, 2);
    el.singleStatus.textContent = "تم ✔️";
  } catch (e) {
    el.singleStatus.textContent = "خطأ: " + e.message;
  }
});

// --- Add to Batch ---
el.addToBatch.addEventListener("click", () => {
  const r = readSingleRaw();
  if (!r.ok) {
    el.singleStatus.textContent = r.err;
    return;
  }
  appendBatchRow(r.raw);
  el.singleStatus.textContent = "أُضيف للسِجل ✅";
});

// --- Batch Predict ---
el.predictBatch.addEventListener("click", async () => {
  el.batchStatus.textContent = "…جاري التنبؤ للدفعة";
  const rows = tableToBatchArrays();
  if (rows.length === 0) {
    el.batchStatus.textContent = "السِجل فارغ—أضِف صفوف أولاً.";
    return;
  }
  try {
    const res = await fetch(`${API_BASE}/predict_batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ batch: rows }),
    });
    const j = await res.json();
    if (!res.ok) throw new Error(j.detail || "API error");
    // Attach results on the right by adding columns visually:
    // For simplicity, we produce a downloadable CSV of inputs + outputs
    window._lastBatch = { inputs: rows, outputs: j };
    el.batchStatus.textContent = `تم ✔️ — ${j.predictions.length} صف`;
  } catch (e) {
    el.batchStatus.textContent = "خطأ: " + e.message;
  }
});

// --- Clear Batch ---
el.clearBatch.addEventListener("click", () => {
  el.batchBody.innerHTML = "";
  el.batchStatus.textContent = "تم المسح";
  window._lastBatch = null;
});

// --- CSV Upload (optional) ---
el.csvInput.addEventListener("change", (ev) => {
  const file = ev.target.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const text = reader.result.toString();
      // Minimal CSV parser: expects header with the same COLS names (order can be any)
      const lines = text.split(/\r?\n/).filter(Boolean);
      const header = lines.shift().split(",").map((s) => s.trim());
      // map header -> indices
      const mapIdx = COLS.map((c) => {
        const i = header.indexOf(c);
        if (i === -1) throw new Error(`CSV missing column: ${c}`);
        return i;
      });
      // Add each row to table matching order COLS
      for (const line of lines) {
        const parts = line.split(",").map((s) => s.trim());
        const raw = {};
        COLS.forEach((c, j) => (raw[c] = Number(parts[mapIdx[j]])));
        appendBatchRow(raw);
      }
      el.batchStatus.textContent = `تم استيراد ${lines.length} صف`;
    } catch (e) {
      el.batchStatus.textContent = "CSV خطأ: " + e.message;
    } finally {
      el.csvInput.value = "";
    }
  };
  reader.readAsText(file);
});

// --- Download CSV of last batch results ---
el.downloadCsv.addEventListener("click", (ev) => {
  ev.preventDefault();
  const last = window._lastBatch;
  if (!last || !last.inputs || !last.outputs) {
    el.batchStatus.textContent = "لا توجد نتائج بعد للتنزيل.";
    return;
  }
  const preds = last.outputs.predictions || [];
  const probs = last.outputs.probabilities || [];
  const header = COLS.concat(["prediction", "probability"]).join(",");
  const rows = last.inputs.map((arr, i) => {
    const line = [...arr, preds[i], probs[i]].join(",");
    return line;
  });
  const csv = header + "\n" + rows.join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "batch_predictions.csv";
  a.click();
  URL.revokeObjectURL(url);
});
