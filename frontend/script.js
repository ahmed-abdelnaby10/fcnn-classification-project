
const API_BASE = location.origin.replace(/:\d+$/, ':8000');

const $ = (s) => document.querySelector(s);
$("#predictBtn").addEventListener("click", async () => {
  const raw = $("#features").value.trim();
  if (!raw) { $("#status").textContent = "Please enter values."; return; }
  const arr = raw.split(",").map(x => parseFloat(x.trim())).filter(x => !Number.isNaN(x));
  $("#status").textContent = "Predicting...";
  try {
    const r = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ features: arr })
    });
    const j = await r.json();
    $("#result").textContent = JSON.stringify(j, null, 2);
    $("#status").textContent = "";
    window._last = (window._last || []);
    window._last.push({ input: arr, output: j });
  } catch (e) {
    $("#status").textContent = "Server error"
  }
});

$("#downloadBtn").addEventListener("click", () => {
  const rows = (window._last || []).map((o, i) => ({
    index: i+1,
    input: JSON.stringify(o.input),
    prediction: o.output?.prediction,
    probability: o.output?.probability
  }));
  const header = ["index","input","prediction","probability"].join(",");
  const body = rows.map(r => [r.index, `"${r.input}"`, r.prediction, r.probability].join(",")).join("\n");
  const csv = header + "\n" + body;
  const blob = new Blob([csv], {type: "text/csv"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "predictions.csv"; a.click();
  URL.revokeObjectURL(url);
});
