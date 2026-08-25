const API = "/api";
const KIND_ZH = { dialogue: "对白", narration: "旁白", thought: "内心", sfx: "拟声" };
const DIR_ZH = { horizontal: "横排", vertical: "竖排" };
const SRC_ZH = { cv: "几何反推", vlm: "本机视觉", human: "人定" };

const state = {
  settings: null,
  project: null,
  pageId: null,
  selectedId: null,
  view: "source",
  zoom: 1,
  busy: false,
  drag: null,
  showBoxes: true,
  selectedIds: [],
  editingId: null,
  projectList: [],
};

const $ = (id) => document.getElementById(id);

function toast(text) {
  const node = $("toast");
  node.hidden = false;
  node.textContent = text;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { node.hidden = true; }, 2800);
}

function setBusy(on, text) {
  state.busy = on;
  $("busy").hidden = !on;
  if (text) $("busyText").textContent = text;
}

async function api(path, options = {}) {
  const res = await fetch(API + path, {
    ...options,
    headers: options.body instanceof FormData ? options.headers : { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      detail = data.detail || JSON.stringify(data);
    } catch {
      detail = await res.text();
    }
    throw new Error(detail || "请求失败");
  }
  const type = res.headers.get("content-type") || "";
  if (type.includes("application/json")) return res.json();
  return res;
}

function rememberProject(id) {
  if (!id) return;
  try { localStorage.setItem("yizi.projectId", id); } catch { /* ignore */ }
}

function escapeHtml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function dropHasFiles(dt) {
  if (!dt) return false;
  if (dt.files && dt.files.length) return true;
  return [...(dt.items || [])].some((item) => item.kind === "file");
}

function isImageFile(file) {
  return Boolean(file) && (file.type.startsWith("image/") || /\.(png|jpe?g|webp)$/i.test(file.name || ""));
}

function skipDropName(name) {
  const n = String(name || "");
  return !n || n.startsWith(".") || n === "__MACOSX" || /^(thumbs\.db|desktop\.ini)$/i.test(n);
}

async function loadProjectList() {
  state.projectList = await api("/projects");
}

function renderProjectPick() {
  const sel = $("projectPick");
  if (!sel) return;
  const items = [...(state.projectList || [])];
  const current = pid();
  if (current && !items.some((item) => item.id === current)) {
    items.unshift({ id: current, name: state.project?.name || "未命名", pages: pages().length });
  }
  sel.hidden = items.length < 1;
  const html = items.map((item) => {
    const label = `${item.name || "未命名"} · ${item.pages || 0}页`;
    return `<option value="${escapeHtml(item.id)}"${item.id === current ? " selected" : ""}>${escapeHtml(label)}</option>`;
  }).join("");
  if (sel.innerHTML !== html) sel.innerHTML = html;
  if (current) sel.value = current;
}

async function openProject(id) {
  if (!id || id === pid()) return;
  state.project = await api(`/projects/${id}`);
  state.pageId = state.project.pages[0]?.id || null;
  state.selectedId = null;
  state.selectedIds = [];
  state.editingId = null;
  rememberProject(id);
  render();
}

function wantedProjectId() {
  try {
    const q = new URLSearchParams(location.search).get("p");
    if (q) return q;
    return localStorage.getItem("yizi.projectId");
  } catch {
    return null;
  }
}

function pid() { return state.project?.id; }
function pages() { return state.project?.pages || []; }
function page() { return pages().find((item) => item.id === state.pageId) || pages()[0] || null; }
function regions() { return page()?.regions || []; }

async function loadSettings() {
  state.settings = await api("/settings");
  const set = state.settings;
  $("engineChip").textContent = set.api_key_set ? "编剧 API 已接" : "本机写词";
  $("visionChip").textContent = set.lettering_vision_ready ? "本机视觉就绪" : "本机视觉未就绪";
  $("visionChip").classList.toggle("dim", !set.lettering_vision_ready);
  $("letterEngine").textContent = set.api_key_set ? "编剧 API 已接" : "本机写词";
  if ($("setWriterHint")) {
    $("setWriterHint").textContent = set.api_key_set ? "Key 已保存。点「测一下通不通」确认。" : "还没填 Key，生成大纲和写词会用本机草稿。";
  }
  if ($("storyHint")) {
    $("storyHint").textContent = set.api_key_set
      ? "编剧 API 已接。按已有框写词嵌进成页，不生成新页新框。挂了会标明并改用本机草稿。"
      : "没填 Key 也能组装。会先看图，再按已有框用本机草稿填进去。不生成新页新框。";
  }
  $("engineChip").classList.remove("warn");
  $("setLetterVision").value = set.lettering_vision || "qwen3vl_4b";
  if ($("setLetterDetect")) $("setLetterDetect").value = set.lettering_detect || "auto";
  $("setApiBase").value = set.api_base || "";
  $("setWriterModel").value = set.writer_model || "deepseek-chat";
  $("setApiKey").value = set.api_key_set ? set.api_key_masked : "";
  if ($("btnClearApiKey")) $("btnClearApiKey").hidden = !set.api_key_set;
  $("setVisionHint").textContent = set.lettering_vision_note || "";
  if ($("setDetectHint")) $("setDetectHint").textContent = set.lettering_detect_note || "";
}

async function ensureProject() {
  if (state.project) return state.project;
  state.project = await api("/projects", { method: "POST", body: JSON.stringify({ name: "未命名剧本" }) });
  await loadProjectList();
  return state.project;
}

async function refresh(projectLike, options = {}) {
  if (projectLike?.id) state.project = projectLike;
  else if (pid()) state.project = await api(`/projects/${pid()}`);
  if (!state.pageId || !pages().some((item) => item.id === state.pageId)) {
    state.pageId = pages()[0]?.id || null;
  }
  rememberProject(pid());
  render(options);
}

function showWorkspace(on) {
  $("welcome").hidden = on;
  $("workspace").hidden = !on;
}

function coachText() {
  const p = page();
  if (!pages().length) return "把成页或整个文件夹拖进来。写下几句话，点「用这几句话回填全部」。";
  if (!(state.project?.premise || "").trim()) return "页已导入。右边写几句你想讲的，不必完整。";
  if (!(state.project?.expanded || state.project?.outline || "").trim()) return "点「用这几句话回填全部」：先看图、构造情节，再按页填框。";
  if (!regions().length) return "故事有了。点「回填全部成页」会按阅读序检出并填框。";
  if (!p.scene) return "框有了。点「回填全部成页」或先看图再按框写词。";
  if (!regions().some((item) => item.text && !item.skip)) return "还有空框。点「回填全部成页」按页填，或点「按框写词」只写本页。";
  if (p.regions.some((item) => item.overflow)) return "有框装不下。改短一点或拉大框。词改完会写回文字层。";
  return "双击框改词，拖角改大小。Shift 多选，Delete 删框，可合并碎框。";
}

function fillIfIdle(id, value) {
  const node = $(id);
  if (!node || document.activeElement === node) return;
  if (node.value !== value) node.value = value;
}

function setUndoVisible(on) {
  if ($("btnStoryUndo")) $("btnStoryUndo").hidden = !on;
}

async function refreshUndo() {
  if (!pid() || !$("btnStoryUndo")) return null;
  try {
    const prev = await api(`/projects/${pid()}/story/preview`);
    setUndoVisible(Boolean(prev.has_undo));
    return prev;
  } catch {
    return null;
  }
}

function applyWriterMeta(meta) {
  if (!meta) return;
  const failed = Boolean(meta.writer_fallback || meta.fallback || meta.write?.fallback || meta.design?.fallback);
  if (failed) {
    $("engineChip").textContent = "编剧 API 没走成";
    $("engineChip").classList.add("warn");
    $("letterEngine").textContent = "本机草稿";
    if ($("storyHint")) $("storyHint").textContent = meta.hint || "编剧 API 没走成，用的是本机草稿。";
    return;
  }
  $("engineChip").classList.remove("warn");
}

async function seeStalePages(prefix) {
  const prev = await api(`/projects/${pid()}/story/preview`);
  const todo = (prev.pages || []).filter((item) => item.needs_see);
  for (let i = 0; i < todo.length; i += 1) {
    setBusy(true, `${prefix}正在看第 ${i + 1}/${todo.length} 页（${todo[i].name}）…`);
    const data = await api(`/projects/${pid()}/pages/${todo[i].id}/lettering/see`, {
      method: "POST",
      body: JSON.stringify({ force: false }),
    });
    const target = pages().find((item) => item.id === todo[i].id);
    if (target && data.page) Object.assign(target, data.page);
  }
  return prev;
}

function renderStorySteps() {
  const box = $("storySteps");
  if (!box) return;
  const hasIdea = Boolean((state.project?.premise || "").trim());
  const hasPack = Boolean((state.project?.expanded || state.project?.outline || "").trim());
  const hasLines = regions().some((item) => item.text && !item.skip);
  let current = 1;
  if (hasIdea && !hasPack) current = 2;
  else if (hasPack && !hasLines) current = 3;
  else if (hasPack && hasLines) current = 0;
  box.querySelectorAll("li").forEach((item) => {
    item.classList.toggle("on", current === 0 || Number(item.dataset.step) === current);
  });
}

function render(options = {}) {
  const list = pages();
  showWorkspace(Boolean(pid()));
  $("projectName").value = state.project?.name || "";
  renderProjectPick();
  $("pageCount").textContent = list.length ? `${list.length} 页` : "";
  if ($("btnDeleteProject")) $("btnDeleteProject").hidden = !pid();
  $("letterHint").textContent = coachText();
  fillIfIdle("letterOutline", state.project?.outline || "");
  fillIfIdle("storyPremise", state.project?.premise || "");
  fillIfIdle("storyCast", state.project?.cast || "");
  fillIfIdle("storyTone", state.project?.tone || "");
  fillIfIdle("storyExpanded", state.project?.expanded || "");
  fillIfIdle("storyStyle", state.project?.style || "");
  renderStorySteps();
  renderPages();
  renderScene();
  renderRegions();
  if (options.reloadImage === false) {
    renderLayers();
    drawOverlay();
    return;
  }
  renderImage();
}

function renderPages() {
  const box = $("pageList");
  box.innerHTML = "";
  if (!pages().length) {
    box.innerHTML = '<div class="tiny">还没有成页。点「再导入」或把文件夹拖进来。</div>';
    return;
  }
  pages().forEach((item, index) => {
    const card = document.createElement("div");
    card.className = "page-card" + (item.id === page()?.id ? " active" : "");
    card.draggable = true;
    card.dataset.pageId = item.id;
    const src = `/api/projects/${pid()}/pages/${item.id}/source.png`;
    card.innerHTML = `<span class="drag-handle" aria-hidden="true">⋮⋮</span><img src="${src}" alt="" /><div><strong>${index + 1}. ${escapeHtml(item.name)}</strong><span>${item.regions?.length || 0} 框${item.scene ? " · 已看图" : ""}</span></div><button type="button" class="danger mini icon-del" data-del="${item.id}">删</button>`;
    card.addEventListener("click", (ev) => {
      if (ev.target.closest("[data-del]")) return;
      state.pageId = item.id;
      state.selectedId = null;
      state.selectedIds = [];
      state.editingId = null;
      render();
    });
    card.querySelector("[data-del]").addEventListener("click", (ev) => {
      ev.stopPropagation();
      deletePage(item.id);
    });
    card.addEventListener("dragstart", (ev) => ev.dataTransfer.setData("text/plain", item.id));
    card.addEventListener("dragover", (ev) => ev.preventDefault());
    card.addEventListener("drop", async (ev) => {
      if (dropHasFiles(ev.dataTransfer)) return;
      ev.preventDefault();
      ev.stopPropagation();
      const from = ev.dataTransfer.getData("text/plain");
      const ids = pages().map((p) => p.id);
      const a = ids.indexOf(from);
      const b = ids.indexOf(item.id);
      if (a < 0 || b < 0) return;
      ids.splice(b, 0, ids.splice(a, 1)[0]);
      const data = await api(`/projects/${pid()}/pages/order`, { method: "POST", body: JSON.stringify({ page_ids: ids }) });
      await refresh(data);
    });
    box.appendChild(card);
  });
}

function renderScene() {
  const p = page();
  const scene = p?.scene;
  const focus = document.activeElement?.id || "";
  if (!["scWhere", "scWho", "scDoing", "scMood", "scJump", "scIntimacy", "scSummary", "scNotes"].includes(focus)) {
    $("scWhere").value = scene?.where || "";
    $("scWho").value = scene?.who || "";
    $("scDoing").value = scene?.doing || "";
    $("scMood").value = scene?.mood || "";
    $("scJump").value = scene?.jump_from_prev || "";
    $("scIntimacy").value = scene?.intimacy || "none";
    $("scSummary").value = p?.page_summary || "";
    if ($("scNotes")) $("scNotes").value = scene?.notes || "";
  }
  $("letterSceneSrc").textContent = SRC_ZH[scene?.source] || "还没看图";
}

function renderRegions() {
  const box = $("letterRegions");
  box.innerHTML = "";
  const items = regions();
  const kinds = {};
  for (const item of items) kinds[item.kind] = (kinds[item.kind] || 0) + 1;
  const brief = ["dialogue", "narration", "thought", "sfx"].filter((kind) => kinds[kind]).map((kind) => `${KIND_ZH[kind]}${kinds[kind]}`).join(" ");
  $("letterCount").textContent = brief ? `${items.length} 框 · ${brief}` : `${items.length} 框`;
  if (!items.length) box.innerHTML = '<div class="tiny">还没有框。点「检出」或「手加」。画布上可拖框。</div>';
  items.forEach((region, index) => {
    const card = document.createElement("div");
    const picked = state.selectedIds.includes(region.id) || region.id === state.selectedId;
    card.className = "letter-card"
      + (picked ? " active" : "")
      + (region.overflow ? " overflow" : "")
      + (region.skip ? " skip" : "");
    const flags = [KIND_ZH[region.kind] || region.kind, DIR_ZH[region.direction] || region.direction, `最多${region.max_chars}字`];
    if (region.skip) flags.push("不替换");
    if (region.locked) flags.push("锁");
    if (region.overflow) flags.push("溢出");
    card.innerHTML = `<div class="letter-card-main"><strong>${index + 1}. ${escapeHtml(region.speaker || "未点名")}</strong><span>${flags.join(" · ")}</span><em>${escapeHtml(region.text || "还没写")}</em></div><button type="button" class="danger mini icon-del" data-del="${region.id}">删</button>`;
    card.addEventListener("click", (ev) => {
      if (ev.target.closest("[data-del]")) return;
      selectRegion(region.id, { additive: ev.shiftKey });
    });
    card.querySelector("[data-del]").addEventListener("click", (ev) => {
      ev.stopPropagation();
      deleteRegions([region.id]);
    });
    box.appendChild(card);
  });
  const current = items.find((item) => item.id === state.selectedId);
  $("letterEditor").hidden = !current;
  if (current) fillEditor(current);
}

function fillEditor(region) {
  if (!region) return;
  const typingSame = document.activeElement?.id === "lrText" && state._filledId === region.id;
  if (typingSame) return;
  state._filledId = region.id;
  $("lrText").value = region.text || "";
  $("lrShort").value = region.text_short || "";
  $("lrSpeaker").value = region.speaker || "";
  $("lrKind").value = region.kind;
  $("lrDir").value = region.direction;
  $("lrSkip").checked = Boolean(region.skip);
  $("lrLock").checked = Boolean(region.locked);
  $("lrX").value = Math.round(region.x);
  $("lrY").value = Math.round(region.y);
  $("lrW").value = Math.round(region.w);
  $("lrH").value = Math.round(region.h);
  $("lrMeta").textContent = `最多约 ${region.max_chars} 字${region.overflow ? " · 当前溢出" : ""}`;
}

function letteringLayer(role) {
  return (page()?.layers || []).find((item) => (item.extra || {}).lettering === role) || null;
}

function setView(view, options = {}) {
  state.view = view;
  if (options.showBoxes) state.showBoxes = true;
  $("btnViewSource").classList.toggle("active", view === "source");
  $("btnViewPreview").classList.toggle("active", view === "preview");
}

function selectRegion(id, options = {}) {
  if (!id) {
    state.selectedId = null;
    state.selectedIds = [];
    state.editingId = null;
    renderRegions();
    drawOverlay();
    return;
  }
  if (options.additive) {
    const set = new Set(state.selectedIds.length ? state.selectedIds : (state.selectedId ? [state.selectedId] : []));
    if (set.has(id)) set.delete(id);
    else set.add(id);
    state.selectedIds = [...set];
    state.selectedId = id;
  } else {
    state.selectedIds = [id];
    state.selectedId = id;
  }
  const region = regions().find((item) => item.id === state.selectedId);
  if (region) fillEditor(region);
  renderRegions();
  drawOverlay();
}

async function updateLayers(pageIds, options = {}) {
  const ids = pageIds && pageIds.length ? pageIds : (page() ? [page().id] : []);
  const erase = options.erase !== false;
  for (const id of ids) {
    if (erase) await api(`/projects/${pid()}/pages/${id}/lettering/erase`, { method: "POST" });
    await api(`/projects/${pid()}/pages/${id}/lettering/typeset`, { method: "POST" });
  }
}

async function refreshLayersAfterEdit(options = {}) {
  if (!pid() || !page()) return;
  await updateLayers([page().id], { erase: Boolean(options.erase) });
  setView("source", { showBoxes: true });
  await refresh();
}

async function showRewritePreview(pageIds, message) {
  await updateLayers(pageIds);
  setView("source", { showBoxes: true });
  await refresh();
  toast(message || "图层已更新。框还能拖，词还能改。");
}

function renderLayerPanel() {
  const box = $("layerList");
  if (!box) return;
  const rows = [
    { id: "source", name: "原页", ready: true, on: true, locked: true, hint: "底" },
    { id: "erase", name: "擦除层", layer: letteringLayer("erase") },
    { id: "text", name: "文字层", layer: letteringLayer("text") },
    { id: "boxes", name: "框", ready: true, on: state.showBoxes !== false, hint: state.showBoxes === false ? "隐藏" : "显示" },
  ];
  box.innerHTML = "";
  rows.forEach((item) => {
    const ready = item.ready || Boolean(item.layer);
    const on = item.locked ? true : (item.id === "boxes" ? state.showBoxes !== false : Boolean(item.layer?.visible));
    const row = document.createElement("button");
    row.type = "button";
    row.className = "layer-row" + (on ? " on" : "") + (ready ? "" : " empty");
    row.innerHTML = `<i>${on ? "●" : "○"}</i><span>${item.name}</span><em>${item.hint || (ready ? (on ? "显示" : "隐藏") : "还没有")}</em>`;
    row.disabled = Boolean(item.locked) || !ready;
    row.addEventListener("click", () => toggleLayer(item.id));
    box.appendChild(row);
  });
}

async function toggleLayer(id) {
  if (id === "boxes") {
    state.showBoxes = state.showBoxes === false;
    drawOverlay();
    renderLayerPanel();
    return;
  }
  const layer = letteringLayer(id);
  if (!layer || !pid() || !page()) return;
  const data = await api(`/projects/${pid()}/pages/${page().id}/layers/${id}`, {
    method: "PATCH",
    body: JSON.stringify({ visible: !layer.visible }),
  });
  layer.visible = data.visible;
  renderLayers();
  if (state.view === "preview") renderImage();
}

function renderLayers() {
  const p = page();
  const bust = encodeURIComponent(state.project?.updated_at || p?.id || "");
  ["erase", "text"].forEach((role) => {
    const img = $(role === "erase" ? "layerErase" : "layerText");
    if (!img) return;
    const layer = letteringLayer(role);
    const show = state.view === "source" && Boolean(layer?.visible);
    if (!p || !layer || !show) {
      img.hidden = true;
      return;
    }
    const next = `/api/projects/${pid()}/pages/${p.id}/layers/${role}.png?t=${bust}`;
    img.onerror = () => { img.hidden = true; };
    if (img.getAttribute("src") !== next) img.src = next;
    img.hidden = false;
  });
  const overlay = $("letterOverlay");
  if (overlay) overlay.hidden = state.showBoxes === false;
  renderLayerPanel();
}

function renderImage() {
  const p = page();
  const img = $("pageImage");
  if (!p || !pid()) {
    img.removeAttribute("src");
    $("letterOverlay").innerHTML = "";
    renderLayers();
    return;
  }
  const bust = encodeURIComponent(state.project?.updated_at || p.id);
  const next = state.view === "preview"
    ? `/api/projects/${pid()}/pages/${p.id}/preview.png?t=${bust}`
    : `/api/projects/${pid()}/pages/${p.id}/source.png?t=${bust}`;
  if (img.getAttribute("src") === next) {
    fitStage();
    renderLayers();
    drawOverlay();
    return;
  }
  img.onload = () => { fitStage(); renderLayers(); drawOverlay(); };
  img.src = next;
}

function fitStage() {
  const p = page();
  const host = $("stageHost");
  const wrap = $("stageWrap");
  if (!p || !wrap.clientWidth) return;
  const scale = Math.min(1.4, (wrap.clientWidth - 48) / p.width, (wrap.clientHeight - 48) / p.height);
  state.zoom = Math.max(0.2, scale);
  host.style.width = `${p.width * state.zoom}px`;
  host.style.height = `${p.height * state.zoom}px`;
  $("pageImage").style.width = "100%";
  $("zoomLabel").textContent = `${Math.round(state.zoom * 100)}%`;
  drawOverlay();
}

function drawOverlay() {
  const overlay = $("letterOverlay");
  const p = page();
  if (state.editingId && document.activeElement?.classList.contains("letter-box-edit")) return;
  overlay.innerHTML = "";
  overlay.hidden = state.showBoxes === false;
  if (!p || overlay.hidden) return;
  const textOn = Boolean(letteringLayer("text")?.visible);
  const picked = new Set(state.selectedIds.length ? state.selectedIds : (state.selectedId ? [state.selectedId] : []));
  regions().forEach((region) => {
    const node = document.createElement("div");
    const on = picked.has(region.id);
    node.className = "letter-box"
      + (on ? " active" : "")
      + (region.overflow ? " overflow" : "")
      + (region.skip ? " skip" : "")
      + (textOn ? " ghost" : "");
    node.dataset.regionId = region.id;
    node.style.left = `${region.x * state.zoom}px`;
    node.style.top = `${region.y * state.zoom}px`;
    node.style.width = `${region.w * state.zoom}px`;
    node.style.height = `${region.h * state.zoom}px`;
    if (state.editingId === region.id) {
      const edit = document.createElement("textarea");
      edit.className = "letter-box-edit";
      edit.value = region.text || "";
      edit.addEventListener("pointerdown", (ev) => ev.stopPropagation());
      edit.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter" && !ev.shiftKey) {
          ev.preventDefault();
          commitInlineEdit(region.id, edit.value);
        }
        if (ev.key === "Escape") {
          ev.preventDefault();
          state.editingId = null;
          drawOverlay();
        }
      });
      edit.addEventListener("blur", () => commitInlineEdit(region.id, edit.value));
      node.appendChild(edit);
      requestAnimationFrame(() => edit.focus());
    } else if (!textOn || on || region.skip) {
      const label = document.createElement("span");
      label.className = "letter-box-label";
      label.textContent = region.skip ? "不替换" : region.text || KIND_ZH[region.kind] || "";
      node.appendChild(label);
    }
    if (on && state.editingId !== region.id) {
      ["nw", "ne", "sw", "se"].forEach((name) => {
        const handle = document.createElement("i");
        handle.className = `letter-handle ${name}`;
        handle.dataset.handle = name;
        node.appendChild(handle);
      });
      const del = document.createElement("button");
      del.type = "button";
      del.className = "danger mini letter-box-del";
      del.dataset.del = region.id;
      del.textContent = "删";
      del.addEventListener("pointerdown", (ev) => ev.stopPropagation());
      del.addEventListener("click", (ev) => {
        ev.stopPropagation();
        deleteRegions([region.id]);
      });
      node.appendChild(del);
    }
    overlay.appendChild(node);
  });
}

async function commitInlineEdit(id, text) {
  if (state.editingId !== id) return;
  state.editingId = null;
  const region = regions().find((item) => item.id === id);
  if (!region) return;
  const next = String(text || "").trim();
  if (next === (region.text || "").trim()) {
    drawOverlay();
    return;
  }
  region.text = next;
  region.locked = true;
  region.skip = false;
  await api(`/projects/${pid()}/pages/${page().id}/regions/${id}`, {
    method: "PATCH",
    body: JSON.stringify({ text: next, locked: true, skip: false }),
  });
  state.editingId = null;
  await refreshLayersAfterEdit({ erase: true });
  toast("词已改，文字层已更新。");
}

function pagePoint(ev) {
  const rect = $("stageHost").getBoundingClientRect();
  return { x: (ev.clientX - rect.left) / state.zoom, y: (ev.clientY - rect.top) / state.zoom };
}

async function work(label, fn) {
  setBusy(true, label);
  try {
    await fn();
  } catch (err) {
    toast(err.message || String(err));
  } finally {
    setBusy(false);
  }
}

async function readAllEntries(reader) {
  const all = [];
  for (;;) {
    const batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
    if (!batch.length) break;
    all.push(...batch);
  }
  return all;
}

async function walkEntry(entry, out, prefix) {
  if (!entry || skipDropName(entry.name)) return;
  if (entry.isFile) {
    const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
    if (!isImageFile(file)) return;
    out.push({ file, path: prefix ? `${prefix}/${entry.name}` : entry.name });
    return;
  }
  if (!entry.isDirectory) return;
  const kids = await readAllEntries(entry.createReader());
  const next = prefix ? `${prefix}/${entry.name}` : entry.name;
  for (const kid of kids) await walkEntry(kid, out, next);
}

function filesFromFileList(fileList) {
  return [...(fileList || [])]
    .filter((file) => isImageFile(file) && !skipDropName(file.name))
    .map((file) => ({ file, path: file.webkitRelativePath || file.name }));
}

function mergePacked(first, second) {
  const seen = new Set();
  const out = [];
  for (const item of [...first, ...second]) {
    const key = `${item.file.size}:${item.file.lastModified}:${item.file.name}:${item.path}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(item);
  }
  return out;
}

async function collectDroppedFiles(dt) {
  const fromFiles = filesFromFileList(dt.files);
  const entries = [];
  for (const item of [...(dt.items || [])]) {
    const entry = item.webkitGetAsEntry?.();
    if (entry) entries.push(entry);
  }
  const fromEntries = [];
  for (const entry of entries) await walkEntry(entry, fromEntries, "");
  return mergePacked(fromEntries, fromFiles);
}

function packImportList(fileList) {
  return [...(fileList || [])].map((item) => {
    if (item && item.file) return { file: item.file, path: item.path || item.file.webkitRelativePath || item.file.name };
    return { file: item, path: (item && (item.webkitRelativePath || item.name)) || "" };
  }).filter((item) => item.file && isImageFile(item.file) && !skipDropName(item.file.name));
}

async function importFromDrop(ev) {
  ev.preventDefault();
  const packed = await collectDroppedFiles(ev.dataTransfer);
  return importFiles(packed);
}

async function importFiles(fileList) {
  const packed = packImportList(fileList);
  packed.sort((a, b) => String(a.path).replace(/\\/g, "/").localeCompare(String(b.path).replace(/\\/g, "/"), "en", { numeric: true, sensitivity: "base" }));
  if (!packed.length) { toast("没有读到图片。可把整个文件夹拖进来。"); return; }
  if (packed.length > 300) { toast("一次最多 300 页。先拆开再拖。"); return; }
  await work(`正在导入 ${packed.length} 页…`, async () => {
    await ensureProject();
    const body = new FormData();
    packed.forEach(({ file, path }) => body.append("files", file, String(path || file.name).replace(/\\/g, "/")));
    const data = await api(`/projects/${pid()}/import`, { method: "POST", body });
    await loadProjectList();
    await refresh(data.project);
    toast(`已导入 ${data.added.length} 页。拖卡片可改阅读序。`);
  });
}

async function runLettering(scope) {
  if (!pid()) return;
  if (scope === "all") {
    await backfillAll({ requirePremise: false, design: true, compose: true });
    return;
  }
  await work("正在处理本页工程…", async () => {
    const data = await api(`/projects/${pid()}/lettering/run`, {
      method: "POST",
      body: JSON.stringify({ scope, page_id: page()?.id, replace: true }),
    });
    setView("source");
    await refresh(data.project);
    toast("本页工程已铺好，不是成品。改框改词后点「更新图层」。");
  });
}

async function writeStoryNow(scope, force = false) {
  if (scope === "all") {
    const list = pages();
    let last = null;
    for (let i = 0; i < list.length; i += 1) {
      setBusy(true, `正在按框写第 ${i + 1}/${list.length} 页（${list[i].name}）…`);
      last = await api(`/projects/${pid()}/lettering/write`, {
        method: "POST",
        body: JSON.stringify({ scope: "page", page_id: list[i].id, force }),
      });
      if (last.project) state.project = last.project;
    }
    await refresh(last?.project);
    const hint = last?.meta?.hint ? `${last.meta.hint} 框还能继续改。` : "已按页写完全部成页。双击框可改。";
    await showRewritePreview(page() ? [page().id] : [], hint);
    return last;
  }
  const data = await api(`/projects/${pid()}/lettering/write`, {
    method: "POST",
    body: JSON.stringify({ scope, page_id: page()?.id, region_id: state.selectedId, force }),
  });
  await refresh(data.project);
  const hint = data.meta?.hint ? `${data.meta.hint} 框还能继续改。` : "词已写好。双击框可改，改完写回文字层。";
  await showRewritePreview(page() ? [page().id] : [], hint);
  return data;
}

async function persistStory() {
  if (!pid()) return;
  await api(`/projects/${pid()}`, {
    method: "PATCH",
    body: JSON.stringify({
      premise: $("storyPremise").value,
      cast: $("storyCast").value,
      tone: $("storyTone").value,
      style: $("storyStyle").value,
      expanded: $("storyExpanded").value,
      outline: $("letterOutline").value,
    }),
  });
}

async function backfillAll(options = {}) {
  const requirePremise = options.requirePremise === true;
  const design = options.design !== false;
  const compose = options.compose !== false;
  if (!pid()) return;
  if (requirePremise && !$("storyPremise").value.trim()) {
    toast("先写几句你想讲的");
    return;
  }
  await work("正在按页回填全部成页…", async () => {
    await persistStory();
    if (compose) {
      await api(`/projects/${pid()}/story/snapshot`, { method: "POST" });
      setUndoVisible(true);
    }
    for (let i = 0; i < pages().length; i += 1) {
      const item = pages()[i];
      if ((item.regions || []).length) continue;
      setBusy(true, `回填：正在检出第 ${i + 1}/${pages().length} 页框…`);
      const detected = await api(`/projects/${pid()}/pages/${item.id}/lettering/detect`, {
        method: "POST",
        body: JSON.stringify({ replace: true }),
      });
      if (detected.page) Object.assign(item, detected.page);
    }
    await seeStalePages("回填：");
    if (design) {
      setBusy(true, "正在按已有框构造全书情节…");
      const designed = await api(`/projects/${pid()}/story/design`, {
        method: "POST",
        body: JSON.stringify({ detect_missing: false, see_missing: false }),
      });
      applyWriterMeta(designed.meta);
      await refresh(designed.project, { reloadImage: false });
    }
    const list = pages();
    let last = null;
    for (let i = 0; i < list.length; i += 1) {
      setBusy(true, `正在回填第 ${i + 1}/${list.length} 页（${list[i].name}）…`);
      last = await api(`/projects/${pid()}/lettering/backfill`, {
        method: "POST",
        body: JSON.stringify({
          page_id: list[i].id,
          detect: true,
          see: true,
          write: true,
          compose,
          force: true,
        }),
      });
      if (last.project) await refresh(last.project, { reloadImage: list[i].id === page()?.id });
    }
    applyWriterMeta(last?.meta);
    await showRewritePreview(page() ? [page().id] : [], last?.meta?.hint || "已回填全部成页。词还能改。");
  });
}

async function writeScope(scope, force = false) {
  if (!pid()) return;
  await work(force ? "正在重写…" : "正在写词…", async () => {
    await writeStoryNow(scope, force);
  });
}

async function saveRegion() {
  const region = regions().find((item) => item.id === state.selectedId);
  if (!region) return;
  await api(`/projects/${pid()}/pages/${page().id}/regions/${region.id}`, {
    method: "PATCH",
    body: JSON.stringify({
      text: $("lrText").value,
      text_short: $("lrShort").value,
      speaker: $("lrSpeaker").value,
      kind: $("lrKind").value,
      direction: $("lrDir").value,
      skip: $("lrSkip").checked,
      locked: true,
      x: Number($("lrX").value),
      y: Number($("lrY").value),
      w: Number($("lrW").value),
      h: Number($("lrH").value),
    }),
  });
  state.editingId = null;
  await refreshLayersAfterEdit({ erase: true });
  toast("已写下。文字层已更新，重写时不会盖掉。");
}

async function deleteRegions(ids) {
  const targets = (ids && ids.length) ? [...ids] : (state.selectedIds.length ? [...state.selectedIds] : (state.selectedId ? [state.selectedId] : []));
  if (!targets.length || !confirm(targets.length > 1 ? `删掉这 ${targets.length} 个框？` : "删掉这一框？")) return;
  await work("正在删框…", async () => {
    for (const id of targets) {
      await api(`/projects/${pid()}/pages/${page().id}/regions/${id}`, { method: "DELETE" });
    }
    state.selectedId = null;
    state.selectedIds = [];
    await refreshLayersAfterEdit({ erase: true });
  });
}

async function deleteSelectedRegions() {
  return deleteRegions();
}

async function clearAllPages() {
  const items = pages();
  if (!items.length) { toast("还没有成页"); return; }
  if (!confirm(`删掉全部 ${items.length} 页？还可以再导入。`)) return;
  await work("正在清空页…", async () => {
    let data = null;
    for (const item of items) {
      data = await api(`/projects/${pid()}/pages/${item.id}`, { method: "DELETE" });
    }
    state.pageId = null;
    state.selectedId = null;
    state.selectedIds = [];
    state.editingId = null;
    await loadProjectList();
    await refresh(data);
    await refreshUndo();
    toast("页已清空。还可以再导入。");
  });
}

async function clearAllRegions() {
  const items = regions();
  if (!items.length || !page()) { toast("还没有框"); return; }
  if (!confirm(`清空本页全部 ${items.length} 个框？`)) return;
  await work("正在清空框…", async () => {
    for (const item of items) {
      await api(`/projects/${pid()}/pages/${page().id}/regions/${item.id}`, { method: "DELETE" });
    }
    state.selectedId = null;
    state.selectedIds = [];
    await refreshLayersAfterEdit({ erase: true });
  });
  toast("本页框已清空。");
}

async function clearStory() {
  if (!pid()) return;
  if (!confirm("清空情节、人物、大纲？页和框还在。")) return;
  await refresh(await api(`/projects/${pid()}`, {
    method: "PATCH",
    body: JSON.stringify({
      premise: "",
      cast: "",
      tone: "",
      style: "",
      expanded: "",
      outline: "",
    }),
  }));
  toast("情节已清空。");
}

async function clearApiKey() {
  if (!state.settings?.api_key_set) { toast("还没存 Key"); return; }
  if (!confirm("清掉编剧 Key？之后用本机草稿。")) return;
  await api("/settings", { method: "PUT", body: JSON.stringify({ api_key: "" }) });
  await loadSettings();
  toast("Key 已清掉。现在用本机草稿。");
}

async function deletePage(pageId) {
  const item = pages().find((page) => page.id === pageId);
  if (!item || !confirm(`删掉「${item.name}」这一页？`)) return;
  await work("正在删页…", async () => {
    const data = await api(`/projects/${pid()}/pages/${pageId}`, { method: "DELETE" });
    if (state.pageId === pageId) {
      state.pageId = null;
      state.selectedId = null;
      state.selectedIds = [];
      state.editingId = null;
    }
    await loadProjectList();
    await refresh(data);
    await refreshUndo();
    toast(pages().length ? "这一页已删。" : "这一页已删。还可以再导入。");
  });
}

async function createBlankProject() {
  await work("正在开新剧本…", async () => {
    const project = await api("/projects", { method: "POST", body: JSON.stringify({ name: "未命名剧本" }) });
    state.project = project;
    state.pageId = null;
    state.selectedId = null;
    state.selectedIds = [];
    rememberProject(project.id);
    await loadProjectList();
    render();
    toast("新剧本已建。导入成页即可。");
  });
}

async function deleteCurrentProject() {
  if (!pid()) return;
  const name = state.project?.name || "这个剧本";
  if (!confirm(`删掉「${name}」整本？页和图层都会去掉。`)) return;
  await work("正在删剧本…", async () => {
    const gone = pid();
    await api(`/projects/${gone}`, { method: "DELETE" });
    if (localStorage.getItem("yizi.projectId") === gone) localStorage.removeItem("yizi.projectId");
    await loadProjectList();
    const next = (state.projectList || []).find((item) => item.id !== gone);
    if (next) {
      await openProject(next.id);
      toast("剧本已删。已切到另一本。");
      return;
    }
    state.project = null;
    state.pageId = null;
    state.selectedId = null;
    state.selectedIds = [];
    render();
    toast("剧本已删。");
  });
}

async function clearSceneCard() {
  if (!pid() || !page()) return;
  if (!confirm("清空本页场景卡？看图结果会丢掉，可以再看。")) return;
  await api(`/projects/${pid()}/pages/${page().id}/scene`, {
    method: "PUT",
    body: JSON.stringify({
      where: "",
      who: "",
      doing: "",
      mood: "",
      jump_from_prev: "",
      intimacy: "none",
      notes: "",
      source: "cv",
    }),
  });
  await api(`/projects/${pid()}/pages/${page().id}`, {
    method: "PATCH",
    body: JSON.stringify({ page_summary: "" }),
  });
  await refresh();
  toast("场景卡已清空。");
}

async function saveScene() {
  if (!pid() || !page()) return;
  await api(`/projects/${pid()}/pages/${page().id}/scene`, {
    method: "PUT",
    body: JSON.stringify({
      where: $("scWhere").value,
      who: $("scWho").value,
      doing: $("scDoing").value,
      mood: $("scMood").value,
      jump_from_prev: $("scJump").value,
      intimacy: $("scIntimacy").value,
      notes: $("scNotes") ? $("scNotes").value : "",
      source: "human",
    }),
  });
  if ($("scSummary").value !== (page().page_summary || "")) {
    await api(`/projects/${pid()}/pages/${page().id}`, {
      method: "PATCH",
      body: JSON.stringify({ page_summary: $("scSummary").value }),
    });
  }
  await refresh();
}

function bindOverlay() {
  const overlay = $("letterOverlay");
  overlay.addEventListener("dblclick", (ev) => {
    const box = ev.target.closest?.(".letter-box");
    if (!box) return;
    ev.preventDefault();
    state.drag = null;
    selectRegion(box.dataset.regionId);
    state.editingId = box.dataset.regionId;
    drawOverlay();
  });
  overlay.addEventListener("pointerdown", (ev) => {
    if (ev.target.closest?.(".letter-box-edit") || ev.target.closest?.(".letter-box-del")) return;
    const handle = ev.target.closest?.("[data-handle]");
    const box = ev.target.closest?.(".letter-box");
    if (!box) { selectRegion(null); return; }
    ev.preventDefault();
    const region = regions().find((item) => item.id === box.dataset.regionId);
    if (!region) return;
    selectRegion(region.id, { additive: ev.shiftKey });
    if (ev.shiftKey) return;
    const pt = pagePoint(ev);
    state.drag = { id: region.id, mode: handle?.dataset.handle || "move", px: pt.x, py: pt.y, x: region.x, y: region.y, w: region.w, h: region.h, moved: false };
    overlay.setPointerCapture?.(ev.pointerId);
  });
  overlay.addEventListener("pointermove", (ev) => {
    if (!state.drag) return;
    const region = regions().find((item) => item.id === state.drag.id);
    if (!region) return;
    const pt = pagePoint(ev);
    const dx = pt.x - state.drag.px;
    const dy = pt.y - state.drag.py;
    if (Math.abs(dx) + Math.abs(dy) > 2) state.drag.moved = true;
    let { x, y, w, h } = state.drag;
    if (state.drag.mode === "move") { x += dx; y += dy; }
    else {
      if (state.drag.mode.includes("w")) { x += dx; w -= dx; }
      if (state.drag.mode.includes("e")) w += dx;
      if (state.drag.mode.includes("n")) { y += dy; h -= dy; }
      if (state.drag.mode.includes("s")) h += dy;
    }
    region.x = Math.max(0, x);
    region.y = Math.max(0, y);
    region.w = Math.max(24, w);
    region.h = Math.max(24, h);
    $("lrX").value = Math.round(region.x);
    $("lrY").value = Math.round(region.y);
    $("lrW").value = Math.round(region.w);
    $("lrH").value = Math.round(region.h);
    drawOverlay();
  });
  overlay.addEventListener("pointerup", async () => {
    if (!state.drag) return;
    const region = regions().find((item) => item.id === state.drag.id);
    const moved = state.drag.moved;
    state.drag = null;
    if (!region || !moved) return;
    await api(`/projects/${pid()}/pages/${page().id}/regions/${region.id}`, {
      method: "PATCH",
      body: JSON.stringify({ x: region.x, y: region.y, w: region.w, h: region.h }),
    });
    await refreshLayersAfterEdit({ erase: true });
  });
}

function bind() {
  const pick = () => $("filePick").click();
  $("btnImport").addEventListener("click", pick);
  $("btnImportHero").addEventListener("click", pick);
  $("btnImportFolder").addEventListener("click", () => $("folderPick").click());
  $("filePick").addEventListener("change", (ev) => { importFiles(ev.target.files); ev.target.value = ""; });
  $("folderPick").addEventListener("change", (ev) => { importFiles(ev.target.files); ev.target.value = ""; });
  ["dropZone", "workspace"].forEach((id) => {
    const node = $(id);
    if (!node) return;
    node.addEventListener("dragover", (ev) => {
      if (!dropHasFiles(ev.dataTransfer)) return;
      ev.preventDefault();
      node.classList.add("over");
    });
    node.addEventListener("dragleave", (ev) => {
      if (!node.contains(ev.relatedTarget)) node.classList.remove("over");
    });
  });
  document.addEventListener("dragover", (ev) => {
    if (!dropHasFiles(ev.dataTransfer)) return;
    ev.preventDefault();
    document.body.classList.add("file-over");
  });
  document.addEventListener("dragleave", (ev) => {
    if (ev.relatedTarget) return;
    document.body.classList.remove("file-over");
  });
  document.addEventListener("drop", (ev) => {
    document.body.classList.remove("file-over");
    if ($("dropZone")) $("dropZone").classList.remove("over");
    if ($("workspace")) $("workspace").classList.remove("over");
    if (!dropHasFiles(ev.dataTransfer)) return;
    ev.preventDefault();
    importFromDrop(ev);
  });
  const demo = () => work("正在打开示例…", async () => {
    const project = await api("/demo", { method: "POST" });
    state.project = project;
    state.pageId = project.pages[0]?.id;
    rememberProject(project.id);
    await loadProjectList();
    render();
    toast("这是两页示例。写几句话，点「用这几句话回填全部」。");
  });
  $("btnDemo").addEventListener("click", demo);
  $("btnDemoHero").addEventListener("click", demo);
  $("projectPick").addEventListener("change", () => openProject($("projectPick").value));
  $("btnNewProject").addEventListener("click", () => createBlankProject());
  $("btnDeleteProject").addEventListener("click", () => deleteCurrentProject());
  $("btnPageAdd").addEventListener("click", () => $("filePick").click());
  $("btnPageAddFolder").addEventListener("click", () => $("folderPick").click());
  $("btnPageClear").addEventListener("click", () => clearAllPages());
  $("btnSceneClear").addEventListener("click", () => clearSceneCard());
  $("btnSceneWrite").addEventListener("click", () => {
    if (!page()?.scene) { toast("先看图或先填场景卡。"); return; }
    writeScope("page", true);
  });
  $("btnStoryClear").addEventListener("click", () => clearStory());
  $("btnRegionsClear").addEventListener("click", () => clearAllRegions());
  $("btnClearApiKey").addEventListener("click", () => clearApiKey());
  $("projectName").addEventListener("change", async () => {
    if (!pid()) return;
    await refresh(await api(`/projects/${pid()}`, { method: "PATCH", body: JSON.stringify({ name: $("projectName").value }) }));
    await loadProjectList();
    renderProjectPick();
  });
  const patchStory = async (body) => {
    if (!pid()) return;
    await refresh(await api(`/projects/${pid()}`, { method: "PATCH", body: JSON.stringify(body) }));
  };
  $("letterOutline").addEventListener("change", () => patchStory({ outline: $("letterOutline").value }));
  $("storyPremise").addEventListener("change", () => patchStory({ premise: $("storyPremise").value }));
  $("storyCast").addEventListener("change", () => patchStory({ cast: $("storyCast").value }));
  $("storyTone").addEventListener("change", () => patchStory({ tone: $("storyTone").value }));
  $("storyExpanded").addEventListener("change", () => patchStory({ expanded: $("storyExpanded").value }));
  $("storyStyle").addEventListener("change", () => patchStory({ style: $("storyStyle").value }));
  const saveStoryForm = () => persistStory();
  $("btnStoryDesign").addEventListener("click", () => {
    if (!pid()) { toast("先导入成页或先写几句"); return; }
    return work("先看需要重看的页，再扩写定调…", async () => {
      await saveStoryForm();
      await seeStalePages("只扩写：");
      setBusy(true, "正在把这几句话扩写并定调…");
      const data = await api(`/projects/${pid()}/story/design`, { method: "POST" });
      applyWriterMeta(data.meta);
      await refresh(data.project);
      toast(data.meta?.hint || "已扩写定调。再点「用这几句话回填全部」按页填框。");
    });
  });
  $("btnStoryAssemble").addEventListener("click", () => backfillAll({ requirePremise: true, design: true, compose: true }));
  $("btnStoryUndo").addEventListener("click", () => {
    if (!pid()) return;
    return work("正在撤销上次组装…", async () => {
      const data = await api(`/projects/${pid()}/story/undo`, { method: "POST" });
      setUndoVisible(false);
      const ids = (data.project?.pages || []).map((item) => item.id);
      await showRewritePreview(ids, data.meta?.hint || "已撤销上次组装。");
    });
  });
  $("btnStoryWritePage").addEventListener("click", () => writeScope("page"));
  $("btnStoryWriteAll").addEventListener("click", () => writeScope("all"));
  $("btnLetterRun").addEventListener("click", () => runLettering("page"));
  $("btnLetterRunAll").addEventListener("click", () => runLettering("all"));
  $("btnLetterDetect").addEventListener("click", () => work("正在检出文字框…", async () => {
    const data = await api(`/projects/${pid()}/pages/${page().id}/lettering/detect`, { method: "POST", body: JSON.stringify({ replace: true }) });
    page().regions = data.page.regions;
    await refresh();
    toast(`检出 ${data.count} 个框。框是草稿，可拖、可删、可手加。`);
  }));
  $("btnLetterAddBox").addEventListener("click", () => work("正在手加一框…", async () => {
    const region = await api(`/projects/${pid()}/pages/${page().id}/regions`, {
      method: "POST",
      body: JSON.stringify({ kind: "dialogue", x: 80, y: 80, w: 180, h: 120 }),
    });
    state.selectedId = region.id;
    await refresh();
  }));
  $("btnLetterSee").addEventListener("click", () => work("本机正在看这一页…", async () => {
    const data = await api(`/projects/${pid()}/pages/${page().id}/lettering/see`, { method: "POST", body: JSON.stringify({ force: false }) });
    page().regions = data.page.regions;
    page().scene = data.page.scene;
    page().see_hint = data.page.see_hint;
    page().page_summary = data.page.page_summary;
    await refresh();
    if (regions().some((item) => !item.skip && !(item.text || "").trim())) {
      setBusy(true, "场景卡好了，正在按已有框写词…");
      await writeStoryNow("page");
      return;
    }
    toast(data.page?.see_hint || "本机已看过这一页，场景卡已更新。");
  }));
  $("btnLetterSeeForce").addEventListener("click", () => work("正在重新看这一页…", async () => {
    const data = await api(`/projects/${pid()}/pages/${page().id}/lettering/see`, { method: "POST", body: JSON.stringify({ force: true }) });
    page().regions = data.page.regions;
    page().scene = data.page.scene;
    page().see_hint = data.page.see_hint;
    page().page_summary = data.page.page_summary;
    await refresh();
    if (regions().some((item) => !item.skip && !(item.text || "").trim())) {
      setBusy(true, "场景卡好了，正在按已有框写词…");
      await writeStoryNow("page");
      return;
    }
    toast(data.page?.see_hint || "已重看，场景卡已更新。");
  }));
  $("btnLetterWrite").addEventListener("click", () => writeScope("page"));
  $("btnLetterWritePage").addEventListener("click", () => writeScope("page", true));
  $("btnLetterWriteAll").addEventListener("click", () => writeScope("all", true));
  $("btnLetterWriteRegion").addEventListener("click", () => {
    if (!state.selectedId) { toast("先点一个框"); return; }
    writeScope("region", true);
  });
  $("btnLetterTypeset").addEventListener("click", () => work("正在更新图层…", async () => {
    await showRewritePreview(page() ? [page().id] : [], "图层已更新。框还能拖，词还能改。");
  }));
  $("btnLetterPrune").addEventListener("click", () => work("正在清碎框…", async () => {
    const data = await api(`/projects/${pid()}/pages/${page().id}/lettering/prune`, { method: "POST" });
    page().regions = data.page.regions;
    state.selectedId = null;
    state.selectedIds = [];
    await refreshLayersAfterEdit({ erase: true });
    const gone = data.meta?.removed || 0;
    const banners = data.meta?.banners || 0;
    const bits = [];
    if (gone) bits.push(`清掉 ${gone} 个碎框`);
    if (banners) bits.push("标题条已标成不替换");
    toast(bits.length ? `${bits.join("，")}。剩下的还能改。` : "没有更多叠在一起的碎框。");
  }));
  $("btnLetterMerge").addEventListener("click", () => {
    const ids = state.selectedIds.length ? state.selectedIds : (state.selectedId ? [state.selectedId] : []);
    if (ids.length < 2) { toast("先按住 Shift 点至少两个框"); return; }
    return work("正在合并选中框…", async () => {
      const data = await api(`/projects/${pid()}/pages/${page().id}/regions/merge`, {
        method: "POST",
        body: JSON.stringify({ region_ids: ids }),
      });
      page().regions = data.page.regions;
      const keep = data.page.regions.find((item) => ids.includes(item.id));
      state.selectedId = keep?.id || null;
      state.selectedIds = state.selectedId ? [state.selectedId] : [];
      await refreshLayersAfterEdit({ erase: true });
      toast("已合成一框。还可以改词、拉大小。");
    });
  });
  $("btnExportPage").addEventListener("click", () => {
    if (!pid() || !page()) return;
    window.location.href = `/api/projects/${pid()}/pages/${page().id}/export.png`;
  });
  $("btnExportLayers").addEventListener("click", () => {
    if (!pid() || !page()) return;
    window.location.href = `/api/projects/${pid()}/pages/${page().id}/layers.zip`;
  });
  $("btnExportZip").addEventListener("click", () => {
    if (!pid()) { toast("先导入成页"); return; }
    window.location.href = `/api/projects/${pid()}/export.zip`;
  });
  $("btnLetterSaveBox").addEventListener("click", saveRegion);
  $("btnLetterDelBox").addEventListener("click", () => deleteSelectedRegions());
  ["scWhere", "scWho", "scDoing", "scMood", "scJump", "scIntimacy", "scSummary", "scNotes"].forEach((id) => {
    if ($(id)) $(id).addEventListener("change", saveScene);
  });
  $("btnViewSource").addEventListener("click", () => { setView("source", { showBoxes: true }); renderImage(); });
  $("btnViewPreview").addEventListener("click", () => {
    if (!letteringLayer("text") && !letteringLayer("erase")) {
      toast("还没有图层。先点「更新图层」。");
    }
    setView("preview", { showBoxes: true });
    renderImage();
  });
  $("btnFit").addEventListener("click", fitStage);
  $("btnSettings").addEventListener("click", async () => { await loadSettings(); $("settingsDlg").showModal(); });
  $("btnSettingsCancel").addEventListener("click", () => $("settingsDlg").close());
  $("settingsForm").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    await api("/settings", {
      method: "PUT",
      body: JSON.stringify({
        lettering_vision: $("setLetterVision").value,
        lettering_detect: $("setLetterDetect") ? $("setLetterDetect").value : "auto",
        api_base: $("setApiBase").value,
        writer_model: $("setWriterModel").value.toLowerCase(),
        api_key: $("setApiKey").value,
      }),
    });
    await loadSettings();
    $("settingsDlg").close();
    toast("设置已保存。");
  });
  $("btnTestWriter").addEventListener("click", () => work("正在测编剧接口…", async () => {
    await api("/settings", {
      method: "PUT",
      body: JSON.stringify({
        api_base: $("setApiBase").value,
        writer_model: $("setWriterModel").value.toLowerCase(),
        api_key: $("setApiKey").value,
      }),
    });
    const data = await api("/settings/writer/test", { method: "POST" });
    $("setWriterHint").textContent = data.hint || (data.ok ? "已连通。" : "没连通。");
    await loadSettings();
    toast(data.hint || (data.ok ? "编剧已连通。" : "没连通。"));
  }));
  $("btnPrepareVision").addEventListener("click", () => work("正在准备本机视觉，第一次会下载模型…", async () => {
    await api("/settings", { method: "PUT", body: JSON.stringify({ lettering_vision: $("setLetterVision").value }) });
    const data = await api("/settings/vision/prepare", { method: "POST" });
    state.settings = data;
    $("setVisionHint").textContent = data.lettering_vision_note || "已就绪";
    await loadSettings();
    toast(data.lettering_vision_ready ? "本机视觉已就绪。" : data.lettering_vision_note);
  }));
  $("btnPrepareDetect").addEventListener("click", () => work("正在下载框检出模型…", async () => {
    const data = await api("/settings/detect/prepare", { method: "POST" });
    state.settings = data;
    if ($("setDetectHint")) $("setDetectHint").textContent = data.lettering_detect_note || "已就绪";
    await loadSettings();
    toast(data.lettering_detect_ready ? "YOLO 泡框检出已就绪。" : data.lettering_detect_note);
  }));
  document.addEventListener("keydown", (ev) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(ev.target.tagName)) return;
    if ((ev.key === "Delete" || ev.key === "Backspace") && (state.selectedId || state.selectedIds.length)) {
      ev.preventDefault();
      deleteSelectedRegions();
      return;
    }
    const list = pages();
    const idx = list.findIndex((item) => item.id === page()?.id);
    if (ev.key === "a" || ev.key === "A" || ev.key === "ArrowLeft") {
      if (idx > 0) { state.pageId = list[idx - 1].id; state.selectedId = null; state.selectedIds = []; render(); }
    }
    if (ev.key === "d" || ev.key === "D" || ev.key === "ArrowRight") {
      if (idx >= 0 && idx < list.length - 1) { state.pageId = list[idx + 1].id; state.selectedId = null; state.selectedIds = []; render(); }
    }
    if ((ev.key === "v" || ev.key === "V") && state.selectedId) {
      const region = regions().find((item) => item.id === state.selectedId);
      if (!region) return;
      region.direction = region.direction === "vertical" ? "horizontal" : "vertical";
      $("lrDir").value = region.direction;
      saveRegion();
    }
  });
  window.addEventListener("resize", fitStage);
  bindOverlay();
}

async function boot() {
  bind();
  await loadSettings();
  await loadProjectList();
  const items = state.projectList || [];
  const want = wantedProjectId();
  const pick = items.find((item) => item.id === want) || items[0];
  if (pick) {
    state.project = await api(`/projects/${pick.id}`);
    state.pageId = state.project.pages[0]?.id || null;
    rememberProject(pick.id);
  }
  render();
  await refreshUndo();
}

boot().catch((err) => toast(err.message || String(err)));
