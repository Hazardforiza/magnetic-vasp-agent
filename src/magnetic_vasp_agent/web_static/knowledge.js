"use strict";

const knowledge = {
  scope: "local", libraries: null, items: [], selected: null,
  drafts: new Map(), selectionToken: 0, loading: false, busy: false,
};

function knowledgeScopeLabel(scope) { return scope === "shared" ? "共享库" : "本地库"; }
function knowledgeKey(item) { return `${item.scope}/${item.id}`; }
function knowledgeTags(value) { return String(value || "").split(/[,，\n]/).map((tag) => tag.trim()).filter(Boolean); }
function knowledgeReady() {
  const library = knowledge.libraries?.[knowledge.scope];
  return Boolean(library?.configured && library.available);
}

async function loadKnowledge() {
  knowledge.loading = true;
  try {
    const response = await api("/api/knowledge");
    knowledge.libraries = { local: response.local, shared: response.shared };
    knowledge.items = response.items || [];
    const path = byId("knowledge-shared-path");
    if (document.activeElement !== path && !path.dataset.edited) path.value = response.shared?.path || "";
    message("knowledge-library-message", "");
    renderKnowledgeLibraries();
    renderKnowledgeList();
  } catch (error) {
    message("knowledge-library-message", error.message, "error");
    if (!knowledge.libraries) {
      for (const element of document.querySelectorAll("[data-library-state]")) element.textContent = "读取失败";
      byId("knowledge-list").innerHTML = '<div class="empty-state compact"><p>暂时无法读取资料，请刷新重试。</p></div>';
    }
  } finally { knowledge.loading = false; }
}

function renderKnowledgeLibraries() {
  for (const scope of ["local", "shared"]) {
    const library = knowledge.libraries?.[scope];
    if (!library) continue;
    const state = document.querySelector(`[data-library-state="${scope}"]`);
    state.textContent = !library.configured ? "未配置" : library.available ? "可用" : "无法访问";
    state.className = `status-pill ${!library.configured ? "muted" : library.available ? "good" : "warn"}`;
    document.querySelector(`[data-library-count="${scope}"]`).textContent = library.available
      ? `${library.item_count || 0} 份资料 · ${library.indexed_count || 0} 份已入库`
      : !library.configured ? "配置目录后即可使用" : library.error || "请检查目录权限或网络连接";
    const button = byId(`knowledge-scope-${scope}`);
    button.classList.toggle("active", scope === knowledge.scope);
    button.setAttribute("aria-pressed", String(scope === knowledge.scope));
    button.disabled = knowledge.busy;
  }
  byId("knowledge-add-title").textContent = `添加到${knowledgeScopeLabel(knowledge.scope)}`;
  byId("knowledge-list-title").textContent = knowledge.scope === "shared" ? "共享资料" : "本地资料";
  for (const formId of ["knowledge-import-form", "knowledge-case-form"]) {
    for (const control of byId(formId).elements) control.disabled = !knowledgeReady() || knowledge.busy;
  }
  byId("refresh-knowledge").disabled = knowledge.busy;
  byId("knowledge-shared-save").disabled = knowledge.busy;
  knowledgeUpdateDirtyState();
  knowledgeUpdateDetailAvailability();
}

function renderKnowledgeList() {
  const library = knowledge.libraries?.[knowledge.scope];
  const items = knowledge.items.filter((item) => item.scope === knowledge.scope);
  byId("knowledge-item-count").textContent = `${items.length} 份`;
  if (!library?.configured) {
    byId("knowledge-list").innerHTML = '<div class="empty-state compact"><p>共享库尚未配置</p><small>在上方填写课题组的共享目录。</small></div>';
    return;
  }
  const warning = !library.available ? `<div class="notice bad">${escapeHtml(library.error || "资料目录无法访问，请检查网络或读写权限。")}</div>` : "";
  byId("knowledge-list").innerHTML = warning + (items.length ? items.map((item) => {
    const selected = knowledge.selected && knowledgeKey(knowledge.selected) === knowledgeKey(item);
    const recovery = item.status === "indexed" && library.index_ready === false;
    return `<button type="button" class="list-row knowledge-list-row ${selected ? "selected" : ""}" data-knowledge-item="${escapeHtml(item.id)}" aria-pressed="${Boolean(selected)}"><span class="knowledge-list-copy"><strong>${escapeHtml(item.title || item.filename || item.id)}</strong><span class="row-sub">${item.kind === "case" ? "计算案例" : "文献"}${item.filename ? ` · ${escapeHtml(item.filename)}` : ""}</span></span><span class="status-pill ${recovery ? "warn" : item.status === "indexed" ? "good" : "muted"}">${recovery ? "索引需恢复" : item.status === "indexed" ? "已入库" : "待确认"}</span></button>`;
  }).join("") : library.available ? '<div class="empty-state compact"><p>还没有资料</p><small>导入文献，或记录第一份计算案例。</small></div>' : "");
}

function knowledgeRememberDraft() {
  const form = byId("knowledge-detail-form");
  if (!form || !knowledge.selected) return;
  const data = new FormData(form);
  if (!Object.keys(knowledgeEditedValues() || {}).length) {
    knowledge.drafts.delete(knowledgeKey(knowledge.selected));
    return;
  }
  knowledge.drafts.set(knowledgeKey(knowledge.selected), {
    title: String(data.get("title") || ""), doi: String(data.get("doi") || ""),
    tags: String(data.get("tags") || ""), text: String(data.get("text") || ""),
  });
}

async function selectKnowledgeItem(scope, id) {
  knowledgeRememberDraft();
  const token = ++knowledge.selectionToken;
  try {
    const item = await api(`/api/knowledge/${encodeURIComponent(scope)}/${encodeURIComponent(id)}`);
    if (token !== knowledge.selectionToken) return;
    knowledge.scope = scope;
    knowledge.selected = item;
    renderKnowledgeLibraries();
    renderKnowledgeList();
    renderKnowledgeDetail();
  } catch (error) { if (token === knowledge.selectionToken) toast(error.message, "error"); }
}

function renderKnowledgeDetail() {
  const item = knowledge.selected;
  if (!item) {
    byId("knowledge-detail").innerHTML = '<div class="detail-empty"><div class="empty-glyph" aria-hidden="true">▥</div><h2 id="knowledge-detail-title">选择或导入一份资料</h2><p>先检查解析文字，再确认入库。</p></div>';
    return;
  }
  const draft = knowledge.drafts.get(knowledgeKey(item)) || {
    title: item.title || "", doi: item.doi || "", tags: (item.tags || []).join(", "), text: item.text || "",
  };
  const indexed = item.status === "indexed";
  const isCase = item.kind === "case";
  const pageInfo = item.page_count ? ` · ${escapeHtml(item.page_count)} 页` : "";
  const pagePreview = (item.pages || []).length > 1 ? `<details class="knowledge-pages"><summary>查看原始分页文字（${escapeHtml(item.pages.length)} 页）</summary>${item.pages.map((page) => `<section><h3>第 ${escapeHtml(page.page)} 页</h3><pre>${escapeHtml(page.text)}</pre></section>`).join("")}</details>` : "";
  const snapshots = isCase ? [["project", "关联项目"], ["result", "关联结果"]].filter(([key]) => item.provenance?.[key]) : [];
  const sourcePreview = snapshots.length ? `<details class="knowledge-pages knowledge-provenance"><summary>查看关联来源快照（只读）</summary><p class="knowledge-note">快照保存了关联时的项目或结果摘要，与可编辑的案例笔记分别保留。自动分析不代表科学验证。</p>${snapshots.map(([key, label]) => `<section><h3>${escapeHtml(label)}</h3><pre>${escapeHtml(JSON.stringify(item.provenance[key], null, 2))}</pre></section>`).join("")}</details>` : "";
  const canPublish = item.scope === "local" && indexed;
  const sharedAvailable = knowledge.libraries?.shared?.configured && knowledge.libraries?.shared?.available;
  byId("knowledge-detail").innerHTML = `<div class="detail-heading"><div><h2 id="knowledge-detail-title">${item.kind === "case" ? "案例" : "文献"}预览</h2><p class="detail-kicker">${escapeHtml(knowledgeScopeLabel(item.scope))}${pageInfo}${item.filename ? ` · ${escapeHtml(item.filename)}` : ""}</p></div><span class="status-pill ${indexed ? "good" : "muted"}" data-knowledge-detail-status>${indexed ? "已入库" : "待确认"}</span></div>
    <div class="knowledge-preview-note" data-knowledge-preview-note></div>
    <form id="knowledge-detail-form" class="form-grid">
      <label class="field field-wide"><span>标题</span><input name="title" type="text" maxlength="300" value="${escapeHtml(draft.title)}" required></label>
      <label class="field"><span>DOI（可选）</span><input name="doi" type="text" maxlength="300" value="${escapeHtml(draft.doi)}"></label>
      <label class="field"><span>标签</span><input name="tags" type="text" maxlength="1000" value="${escapeHtml(draft.tags)}" placeholder="用逗号分隔"></label>
      <label class="field field-wide"><span>${isCase ? "案例笔记" : "可检索正文"}</span><textarea name="text" rows="13" required>${escapeHtml(draft.text)}</textarea><small>${isCase ? "案例笔记由你编写，可随时编辑；关联来源快照单独保留，入库时与笔记共同供检索。" : item.page_count ? "修改正文后，原 PDF 页码可能无法继续对应；检索会明确标识。" : "正文应保留必要的条件、参数和出处，避免只记录结论。"}</small></label>
      <div class="knowledge-note field-wide">科学结论未核验 · 入库代表可检索，不代表参数或结论已被验证。</div>
      <div class="form-actions knowledge-detail-actions"><button type="submit" class="button" data-knowledge-save>保存草稿</button><button type="button" class="button button-primary" data-knowledge-index>${indexed ? "确认并更新索引" : "确认入库"}</button>${canPublish ? `<button type="button" class="button" data-knowledge-publish ${!sharedAvailable ? "disabled" : ""}>发布到共享库</button>` : ""}</div>
    </form>${canPublish ? `<p class="knowledge-note" data-knowledge-publish-note ${sharedAvailable ? "hidden" : ""}>配置可用的共享目录后，可将此资料发布到共享库。</p>` : ""}
    <div id="knowledge-detail-message" class="inline-message" role="status" aria-live="polite"></div>${sourcePreview}${pagePreview}`;
  knowledgeUpdateDirtyState();
  knowledgeUpdateDetailAvailability();
}

function knowledgeUpdateDetailAvailability() {
  const item = knowledge.selected;
  const status = document.querySelector("[data-knowledge-detail-status]");
  const note = document.querySelector("[data-knowledge-preview-note]");
  if (!item || !status || !note) return;
  const indexed = item.status === "indexed";
  const recovery = indexed && knowledge.libraries?.[item.scope]?.index_ready === false;
  status.textContent = recovery ? "索引需恢复" : indexed ? "已入库" : "待确认";
  status.className = `status-pill ${recovery ? "warn" : indexed ? "good" : "muted"}`;
  note.textContent = recovery ? "检索索引暂不可用，Agent 当前无法检索这份资料。检查内容后，点击“确认并更新索引”恢复。" : indexed ? "Agent 已可检索这份资料。修改后请重新确认入库。" : "解析已完成。检查并修正文字后，点击“确认入库”让 Agent 可以检索。";
}

function knowledgeEditedValues() {
  const item = knowledge.selected;
  const form = byId("knowledge-detail-form");
  if (!item || !form) return null;
  const data = new FormData(form);
  const current = { title: String(data.get("title") || "").trim(), doi: String(data.get("doi") || "").trim(), tags: knowledgeTags(data.get("tags")), text: String(data.get("text") || "") };
  const changes = {};
  if (current.title !== (item.title || "")) changes.title = current.title;
  if (current.doi !== (item.doi || "")) changes.doi = current.doi;
  if (JSON.stringify(current.tags) !== JSON.stringify(item.tags || [])) changes.tags = current.tags;
  if (current.text !== (item.text || "")) changes.text = current.text;
  return changes;
}

function knowledgeUpdateDirtyState() {
  const dirty = Boolean(Object.keys(knowledgeEditedValues() || {}).length);
  const publish = document.querySelector("[data-knowledge-publish]");
  if (publish) {
    const recovery = knowledge.libraries?.[knowledge.selected?.scope]?.index_ready === false;
    publish.disabled = knowledge.busy || dirty || recovery || !knowledge.libraries?.shared?.available;
    publish.title = recovery ? "请先恢复检索索引" : dirty ? "请先确认修改并更新索引" : "复制已入库资料到共享目录";
  }
  const publishNote = document.querySelector("[data-knowledge-publish-note]");
  if (publishNote) publishNote.hidden = Boolean(knowledge.libraries?.shared?.available);
  const save = document.querySelector("[data-knowledge-save]");
  if (save) save.disabled = knowledge.busy || !dirty;
}

async function knowledgeSaveDetail(event, index = false) {
  event?.preventDefault();
  const form = byId("knowledge-detail-form");
  if (!form?.reportValidity() || knowledge.busy) return;
  const item = knowledge.selected;
  const key = knowledgeKey(item);
  const changes = knowledgeEditedValues();
  const base = `/api/knowledge/${item.scope}/${encodeURIComponent(item.id)}`;
  knowledge.busy = true;
  for (const control of form.elements) control.disabled = true;
  renderKnowledgeLibraries();
  message("knowledge-detail-message", index ? "正在保存并更新索引…" : "正在保存草稿…");
  try {
    let saved = item;
    if (Object.keys(changes).length) {
      saved = await api(base, { method: "PATCH", body: changes });
      if (knowledge.selected && knowledgeKey(knowledge.selected) === key) knowledge.selected = saved;
    }
    if (index) saved = await api(`${base}/index`, { method: "POST", body: {} });
    knowledge.drafts.delete(key);
    if (knowledge.selected && knowledgeKey(knowledge.selected) === key) knowledge.selected = saved;
    await loadKnowledge();
    renderKnowledgeDetail();
    message("knowledge-detail-message", index ? "已入库。现在可在下方测试检索，Agent 也可引用这份资料。" : "已保存草稿，确认入库后可被检索。", "success");
  } catch (error) {
    renderKnowledgeDetail();
    message("knowledge-detail-message", error.message, "error");
  } finally { knowledge.busy = false; renderKnowledgeLibraries(); }
}

async function knowledgePublish() {
  if (knowledge.busy || Object.keys(knowledgeEditedValues() || {}).length) return;
  const item = knowledge.selected;
  knowledge.busy = true;
  renderKnowledgeLibraries();
  const button = document.querySelector("[data-knowledge-publish]");
  if (button) button.disabled = true;
  message("knowledge-detail-message", "正在发布到共享库…");
  try {
    await api(`/api/knowledge/local/${encodeURIComponent(item.id)}/publish`, { method: "POST", body: {} });
    await loadKnowledge();
    message("knowledge-detail-message", "已发布到共享库。本地资料仍保留，可切换到共享库查看。", "success");
  } catch (error) { message("knowledge-detail-message", error.message, "error"); }
  finally { knowledge.busy = false; renderKnowledgeLibraries(); knowledgeUpdateDirtyState(); }
}

async function knowledgeImport(event, kind) {
  event.preventDefault();
  if (!knowledgeReady() || knowledge.busy) return;
  const form = event.currentTarget;
  const data = new FormData(form);
  knowledge.busy = true;
  renderKnowledgeLibraries();
  message("knowledge-import-message", kind === "case" ? "正在整理案例…" : "正在解析文件…");
  try {
    let item;
    if (kind === "case") {
      const body = { scope: knowledge.scope, title: String(data.get("title") || ""), text: String(data.get("text") || ""), tags: knowledgeTags(data.get("tags")) };
      if (data.get("project_id")) body.project_id = data.get("project_id");
      if (data.get("result_id")) body.result_id = data.get("result_id");
      item = await api("/api/knowledge/cases", { method: "POST", body });
    } else {
      data.set("scope", knowledge.scope);
      item = await api("/api/knowledge/import", { method: "POST", body: data });
    }
    knowledgeRememberDraft();
    ++knowledge.selectionToken;
    knowledge.selected = item;
    knowledge.scope = item.scope;
    form.reset();
    await loadKnowledge();
    renderKnowledgeDetail();
    message("knowledge-import-message", "草稿已创建，请在预览中检查文字并确认入库。", "success");
    byId("knowledge-detail-title").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) { message("knowledge-import-message", error.message, "error"); }
  finally { knowledge.busy = false; renderKnowledgeLibraries(); }
}

async function knowledgeSaveShared(event) {
  event.preventDefault();
  if (knowledge.busy) return;
  knowledge.busy = true;
  const button = byId("knowledge-shared-save");
  renderKnowledgeLibraries();
  message("knowledge-shared-message", "正在检查共享目录…");
  try {
    await api("/api/knowledge/shared", { method: "PUT", body: { path: byId("knowledge-shared-path").value.trim() || null } });
    delete byId("knowledge-shared-path").dataset.edited;
    if (knowledge.selected?.scope === "shared") {
      knowledgeRememberDraft();
      knowledge.selected = null;
      ++knowledge.selectionToken;
      renderKnowledgeDetail();
    }
    await loadKnowledge();
    const shared = knowledge.libraries?.shared;
    message("knowledge-shared-message", !shared?.configured ? "已断开共享库，本地库可继续使用。" : shared.available ? "共享库已连接。其他电脑配置同一目录即可共用资料。" : shared.error || "目录已保存，但目前无法访问。", shared?.configured && !shared.available ? "error" : "success");
  } catch (error) { message("knowledge-shared-message", error.message, "error"); }
  finally { knowledge.busy = false; renderKnowledgeLibraries(); button.disabled = false; }
}

async function knowledgeSearch(event) {
  event.preventDefault();
  const data = new FormData(event.currentTarget);
  const button = byId("knowledge-search-submit");
  button.disabled = true;
  message("knowledge-search-message", "正在检索已入库资料…");
  try {
    const result = await api("/api/knowledge/search", { method: "POST", body: { query: String(data.get("query") || ""), scope: data.get("scope"), limit: 5 } });
    const evidence = result.evidence || [];
    message("knowledge-search-message", evidence.length ? `找到 ${evidence.length} 条相关片段。` : "没有找到相关片段。请确认资料已入库，或换用材料名称、方法与参数关键词。", evidence.length ? "success" : "");
    const warnings = (result.warnings || []).map((warning) => `<div class="notice">${escapeHtml(warning)}</div>`).join("");
    byId("knowledge-search-results").innerHTML = warnings + evidence.map((entry, index) => `<article class="knowledge-evidence"><div class="knowledge-evidence-heading"><strong>${index + 1}. ${escapeHtml(entry.title)}</strong><span class="status-pill muted">${escapeHtml(knowledgeScopeLabel(entry.scope))}</span></div><p class="knowledge-citation">${entry.source_segment_label ? `${escapeHtml(entry.source_segment_label)} · ` : ""}${entry.page ? `PDF 第 ${escapeHtml(entry.page)} 页 · ` : ""}${entry.doi ? `DOI: ${escapeHtml(entry.doi)} · ` : ""}科学结论未核验</p><blockquote>${escapeHtml(entry.quote)}</blockquote><button type="button" class="text-button" data-knowledge-evidence-id="${escapeHtml(entry.item_id)}" data-knowledge-evidence-scope="${escapeHtml(entry.scope)}">查看来源</button></article>`).join("");
  } catch (error) {
    message("knowledge-search-message", error.message, "error");
    byId("knowledge-search-results").innerHTML = "";
  } finally { button.disabled = false; }
}

function knowledgeRefreshSeeds() {
  const projects = byId("knowledge-case-project");
  const results = byId("knowledge-case-result");
  const projectValue = projects.value;
  const resultValue = results.value;
  projects.innerHTML = '<option value="">不关联项目</option>' + app.projects.map((project) => `<option value="${escapeHtml(project.id)}">${escapeHtml(project.workflow_id || project.source_name || project.title || project.id)}</option>`).join("");
  results.innerHTML = '<option value="">不关联结果</option>' + app.results.map((result) => `<option value="${escapeHtml(result.id)}">${escapeHtml(result.title || shortId(result.id))}${result.final_energy == null ? "" : ` · ${escapeHtml(formatNumber(result.final_energy))} eV`}</option>`).join("");
  if ([...projects.options].some((option) => option.value === projectValue)) projects.value = projectValue;
  if ([...results.options].some((option) => option.value === resultValue)) results.value = resultValue;
}

function knowledgeEntered() { knowledgeRefreshSeeds(); }

function knowledgeSelectAdd(kind, focus = false) {
  for (const name of ["literature", "case"]) {
    const tab = byId(`knowledge-tab-${name}`);
    tab.classList.toggle("active", name === kind);
    tab.setAttribute("aria-selected", String(name === kind));
    tab.tabIndex = name === kind ? 0 : -1;
    byId(`knowledge-${name}-panel`).hidden = name !== kind;
  }
  message("knowledge-import-message", "");
  if (kind === "case") knowledgeRefreshSeeds();
  if (focus) byId(`knowledge-tab-${kind}`).focus();
}

function knowledgeInit() {
  renderKnowledgeLibraries();
  byId("refresh-knowledge").addEventListener("click", () => { if (!knowledge.busy) loadKnowledge(); });
  byId("view-knowledge").addEventListener("click", (event) => {
    const scope = event.target.closest("[data-knowledge-scope]");
    if (scope && !knowledge.busy) {
      knowledgeRememberDraft();
      knowledge.scope = scope.dataset.knowledgeScope;
      knowledge.selected = null;
      ++knowledge.selectionToken;
      renderKnowledgeLibraries(); renderKnowledgeList(); renderKnowledgeDetail();
      message("knowledge-import-message", "");
      if (knowledge.scope === "shared" && !knowledge.libraries?.shared?.configured) byId("knowledge-shared-settings").open = true;
    }
    const add = event.target.closest("[data-knowledge-add]");
    if (add) knowledgeSelectAdd(add.dataset.knowledgeAdd);
    const item = event.target.closest("[data-knowledge-item]");
    if (item && !knowledge.busy) selectKnowledgeItem(knowledge.scope, item.dataset.knowledgeItem);
    const evidence = event.target.closest("[data-knowledge-evidence-id]");
    if (evidence && !knowledge.busy) selectKnowledgeItem(evidence.dataset.knowledgeEvidenceScope, evidence.dataset.knowledgeEvidenceId);
    if (event.target.closest("[data-knowledge-index]")) knowledgeSaveDetail(event, true);
    if (event.target.closest("[data-knowledge-publish]")) knowledgePublish();
  });
  byId("view-knowledge").addEventListener("keydown", (event) => {
    if (event.target.matches("[data-knowledge-add]") && ["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
      event.preventDefault();
      const kind = event.key === "Home" ? "literature" : event.key === "End" ? "case" : event.target.dataset.knowledgeAdd === "case" ? "literature" : "case";
      knowledgeSelectAdd(kind, true);
    }
  });
  byId("knowledge-import-form").addEventListener("submit", (event) => knowledgeImport(event, "literature"));
  byId("knowledge-case-form").addEventListener("submit", (event) => knowledgeImport(event, "case"));
  byId("knowledge-detail").addEventListener("submit", (event) => { if (event.target.id === "knowledge-detail-form") knowledgeSaveDetail(event); });
  byId("knowledge-detail").addEventListener("input", () => { knowledgeRememberDraft(); knowledgeUpdateDirtyState(); });
  byId("knowledge-shared-form").addEventListener("submit", knowledgeSaveShared);
  byId("knowledge-shared-path").addEventListener("input", (event) => { event.target.dataset.edited = "true"; });
  byId("knowledge-search-form").addEventListener("submit", knowledgeSearch);
}
