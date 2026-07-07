/* finsight UI — upload/sample → page viewer → chat with streamed agent events.
   Citations (page+block) click through to a bbox highlight on the page. */

const $ = (id) => document.getElementById(id);
const state = { doc: null, page: 1, pageCount: 0, ws: null, busy: false,
                lastAnswerEl: null, hlTimer: null, shownPage: null };

const esc = (s) => { const d = document.createElement("div"); d.textContent = s ?? ""; return d.innerHTML; };

/* ── status badge ── */
fetch("/api/status").then(r => r.json()).then(s => {
  const b = $("mode-badge");
  b.textContent = s.cloud ? `cloud · ${s.models.answer.split("/").pop()}` : "offline · extractive";
  b.classList.toggle("cloud", s.cloud);
});

/* ── document loading ── */
$("btn-sample").onclick = () => loadDoc(fetch("/load-sample", { method: "POST" }));
$("file-input").onchange = (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const fd = new FormData(); fd.append("file", f);
  loadDoc(fetch("/upload", { method: "POST", body: fd }));
};

async function loadDoc(promise) {
  const res = await (await promise).json();
  if (res.error) return addMsg("err", res.error);
  state.doc = res.doc_id; state.pageCount = res.page_count; state.page = 1;
  state.shownPage = null;                    // force a fresh render for the new document
  $("doc-name").textContent = `${res.name} · ${res.page_count} pages`;
  if (res.status !== "ready") await waitReady(res.doc_id);
  $("progress").classList.add("hidden");
  showPage(1); connectWs(); setAsk(true);
  addMsg("agent", "Document ready — ask me about it.");
}

async function waitReady(id) {
  const prog = $("progress"), bar = $("progress-bar"), label = $("progress-label");
  prog.classList.remove("hidden");
  for (;;) {
    const s = await (await fetch(`/doc/${id}/status`)).json();
    if (s.status === "error") { prog.classList.add("hidden"); addMsg("err", s.error); throw new Error(s.error); }
    const pct = Math.round(100 * s.parsed_pages / s.page_count);
    bar.style.width = pct + "%";
    label.textContent = `parsing ${s.parsed_pages}/${s.page_count} pages (${s.parser || "auto"})`;
    if (s.status === "ready") return;
    await new Promise(r => setTimeout(r, 700));
  }
}

/* ── page viewer ── */
function showPage(n) {
  if (!state.doc) return;
  const target = Math.min(Math.max(1, n), state.pageCount);
  const img = $("page-img");
  if (state.shownPage !== target) {          // only (re)load when the page actually changes
    state.shownPage = target;
    img.src = `/doc/${state.doc}/page/${target}.png`;
  }
  state.page = target;
  img.style.display = "inline-block";
  $("pg-label").textContent = `${target} / ${state.pageCount}`;
  $("hl").classList.add("hidden");
}
$("pg-prev").onclick = () => showPage(state.page - 1);
$("pg-next").onclick = () => showPage(state.page + 1);

async function highlight(page, blockId) {
  showPage(page);
  const img = $("page-img"), hl = $("hl");
  if (!img.complete)                          // one-shot wait — never a persistent handler
    await new Promise((res) => img.addEventListener("load", res, { once: true }));
  const data = await (await fetch(`/doc/${state.doc}/page/${page}`)).json();
  const blk = (data.blocks || []).find(b => b.id === blockId);
  if (!blk) return;
  const [x0, y0, x1, y1] = blk.bbox;
  const sx = img.clientWidth / data.page_w, sy = img.clientHeight / data.page_h;
  const wrap = $("page-wrap").getBoundingClientRect(), ib = img.getBoundingClientRect();
  const offX = ib.left - wrap.left + $("page-wrap").scrollLeft;
  const offY = ib.top - wrap.top + $("page-wrap").scrollTop;
  Object.assign(hl.style, {
    left: offX + x0 * sx + "px", top: offY + y0 * sy + "px",
    width: (x1 - x0) * sx + "px", height: (y1 - y0) * sy + "px",
  });
  hl.classList.remove("hidden");
  clearTimeout(state.hlTimer);               // a stale timer must not hide a fresh highlight
  state.hlTimer = setTimeout(() => hl.classList.add("hidden"), 3000);
}

/* ── chat ── */
function setAsk(on) { $("q").disabled = !on; $("send").disabled = !on; if (on) $("q").focus(); }

function addMsg(cls, html, raw = false) {
  const el = document.createElement("div");
  el.className = `msg ${cls}`;
  if (raw) el.innerHTML = html; else el.textContent = html;
  $("log").appendChild(el); $("log").scrollTop = $("log").scrollHeight;
  return el;
}

function addChip(row, text, cls = "") {
  const c = document.createElement("span");
  c.className = `chip ${cls}`; c.textContent = text;
  row.appendChild(c); $("log").scrollTop = $("log").scrollHeight;
}

function connectWs() {
  if (state.ws) state.ws.close();
  const proto = location.protocol === "https:" ? "wss" : "ws";
  state.ws = new WebSocket(`${proto}://${location.host}/ws/ask/${state.doc}`);
  let trace = null;
  state.ws.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.type === "error") { addMsg("err", ev.error); state.busy = false; setAsk(true); }
    else if (ev.type === "agent_start") { trace = document.createElement("div"); trace.className = "trace"; $("log").appendChild(trace); }
    else if (ev.type === "agent_node") renderNode(trace, ev);
    else if (ev.type === "agent_answer") renderAnswer(ev);
    else if (ev.type === "agent_done") { state.busy = false; setAsk(true); }
  };
}

function renderNode(trace, ev) {
  if (!trace) return;
  if (ev.node === "supervise") addChip(trace, `lane: ${ev.task}`);
  else if (ev.node === "retrieve") addChip(trace, `retrieve #${ev.attempt} → ${ev.k}`);
  else if (ev.node === "grade") addChip(trace, `grade: ${ev.verdict}`, ev.verdict === "relevant" ? "ok" : "warn");
  else if (ev.node === "rewrite") addChip(trace, `rewrite: ${ev.query}`);
  else if (ev.node === "calculate") addChip(trace, `calc: ${ev.expr} = ${Number(ev.result).toFixed(2)}`, "ok");
  else if (ev.node === "cite_check") {
    const bad = ev.unverified || [];
    addChip(trace, bad.length ? `⚠ unverified: ${bad.join(", ")}` : "✓ figures verified", bad.length ? "warn" : "ok");
    if (ev.answer && state.lastAnswerEl)                       // caveat was appended
      state.lastAnswerEl.querySelector(".ans-text").textContent = ev.answer;
  }
}

function renderAnswer(ev) {
  let html = `<div class="ans-text">${esc(ev.answer)}</div>`;
  if (ev.claims && ev.claims.length) {
    html += `<div class="claims">` + ev.claims.map(cl => {
      const mark = cl.verified === false ? `<span class="x">✗</span>` : `<span class="v">✓</span>`;
      const cites = (cl.citations || []).map(ct =>
        `<button class="cite" data-p="${ct.page}" data-b="${ct.block_id ?? ""}">p${ct.page}${ct.block_id != null ? "·b" + ct.block_id : ""}</button>`).join(" ");
      return `<div class="claim">${mark}<span>${esc(cl.text)}</span>${cites}</div>`;
    }).join("") + `</div>`;
  }
  state.lastAnswerEl = addMsg("agent", html, true);
  state.lastAnswerEl.querySelectorAll(".cite").forEach(btn => {
    btn.onclick = () => highlight(+btn.dataset.p, btn.dataset.b === "" ? null : +btn.dataset.b);
  });
}

$("ask-form").onsubmit = (e) => {
  e.preventDefault();
  const q = $("q").value.trim();
  if (!q || state.busy || !state.ws || state.ws.readyState !== 1) return;
  addMsg("user", q); $("q").value = ""; state.busy = true; setAsk(false);
  state.ws.send(JSON.stringify({ question: q }));
};
