"use strict";

const app = {
  csrfToken: "",
  status: null,
  projects: [],
  results: [],
  selectedProject: null,
  selectedResult: null,
  selectedStep: null,
  selectedFile: "INCAR",
  session: null,
  sessionTimer: null,
};

const viewTitles = {
  projects: "项目",
  agent: "Agent",
  results: "结果",
  settings: "环境",
};

const toolNames = {
  "execution.probe_live": "读取节点状态",
  "execution.topss_snapshot": "读取节点占用",
  "execution.resource_plan": "规划计算资源",
  "execution.resource_grant": "授权计算资源",
  "execution.transfer_plan": "规划文件传输",
  "execution.remote_status": "查询远程状态",
  "execution.fetch": "回传计算结果",
  "execution.smoke": "MPI 连通测试",
  "execution.transfer": "传输输入文件",
  "execution.launch": "启动 VASP 计算",
};

const resultPlotNames = {
  dos: "总态密度 DOS",
  band: "能带结构",
  pdos: "投影态密度 PDOS",
  combined: "能带与态密度",
};

function byId(id) { return document.getElementById(id); }

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
}

function formatNumber(value, digits = 5) {
  const number = Number(value);
  return value === null || value === undefined || !Number.isFinite(number)
    ? "—" : number.toFixed(digits).replace(/\.?0+$/, "");
}

function shortId(value) {
  const text = String(value ?? "");
  return text.length > 22 ? `${text.slice(0, 19)}…` : text;
}

function message(target, text, type = "") {
  const element = byId(target);
  element.textContent = text;
  element.className = `inline-message ${type}`.trim();
}

function toast(text, type = "") {
  const element = byId("toast");
  element.textContent = text;
  element.className = `toast ${type}`.trim();
  element.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { element.hidden = true; }, 4400);
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.method && options.method.toUpperCase() !== "GET") {
    headers.set("X-CSRF-Token", app.csrfToken);
  }
  if (options.body && !(options.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
    options.body = JSON.stringify(options.body);
  }
  let response;
  try {
    response = await fetch(path, { credentials: "same-origin", ...options, headers });
  } catch {
    throw new Error("无法连接本机服务，请检查服务是否正在运行。");
  }
  const text = await response.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch { data = { detail: text }; }
  if (!response.ok) {
    const detail = data.detail || data.error || data.message || `请求失败 (${response.status})`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

function setView(name) {
  const view = viewTitles[name] ? name : "projects";
  for (const [key] of Object.entries(viewTitles)) {
    const section = byId(`view-${key}`);
    section.hidden = key !== view;
    section.classList.toggle("active", key === view);
  }
  document.querySelectorAll(".nav-link").forEach((link) => {
    const active = link.dataset.view === view;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  byId("topbar-title").textContent = viewTitles[view];
  byId("sidebar").classList.remove("open");
  byId("menu-toggle").setAttribute("aria-expanded", "false");
}

function currentView() {
  return location.hash.replace(/^#/, "") || "projects";
}

async function loadStatus() {
  try {
    const status = await api("/api/status");
    app.status = status;
    app.csrfToken = status.csrf_token || "";
    byId("sidebar-status").textContent = status.ready ? "本机服务已连接" : "服务未就绪";
    byId("sidebar-status-dot").className = `status-dot ${status.ready ? "ok" : "error"}`;
    byId("agent-config-state").textContent = status.agent_configured ? "模型已配置" : "模型未配置";
    byId("agent-submit").disabled = !status.agent_configured;
    byId("project-submit").disabled = !status.project_generation_ready;
    byId("environment-list").innerHTML = [
      settingRow("本机服务", status.ready, "已连接", "未就绪"),
      settingRow("输入生成", status.project_generation_ready, "已就绪", "需要配置"),
      settingRow("Agent 模型", status.agent_configured, "已配置", "需要配置"),
    ].join("");
    byId("service-origin").textContent = location.origin;
    if (!status.agent_configured) message("agent-form-message", "模型配置尚未就绪。", "error");
    else message("agent-form-message", "");
    if (!status.project_generation_ready) message("project-form-message", "输入生成环境尚未就绪。", "error");
    else message("project-form-message", "");
  } catch (error) {
    byId("sidebar-status").textContent = "连接失败";
    byId("sidebar-status-dot").className = "status-dot error";
    byId("environment-list").innerHTML = `<div class="notice bad">${escapeHtml(error.message)}</div>`;
    byId("agent-submit").disabled = true;
    byId("project-submit").disabled = true;
    throw error;
  }
}

function settingRow(label, ready, yes, no) {
  return `<div class="setting-row"><span>${escapeHtml(label)}</span><strong><span class="status-pill ${ready ? "good" : "warn"}">${escapeHtml(ready ? yes : no)}</span></strong></div>`;
}

function projectState(project) {
  if (project.decision === "accept") return { label: "已接受", tone: "good" };
  if (project.decision === "reject") return { label: "已拒绝", tone: "bad" };
  if (project.decision === "revise") return { label: "需修改", tone: "warn" };
  return { label: "待审核", tone: "warn" };
}

async function loadProjects(preferredId = app.selectedProject?.id) {
  const list = byId("project-list");
  list.innerHTML = '<div class="loading-line">正在读取项目...</div>';
  try {
    const data = await api("/api/projects");
    app.projects = data.projects || [];
    byId("project-count").textContent = `${app.projects.length} 个项目`;
    renderProjectList();
    if (preferredId && app.projects.some((item) => item.id === preferredId)) {
      await selectProject(preferredId);
    } else if (app.selectedProject && !app.projects.some((item) => item.id === app.selectedProject.id)) {
      app.selectedProject = null;
      renderProjectDetail();
    }
  } catch (error) {
    list.innerHTML = `<div class="notice bad">${escapeHtml(error.message)}</div>`;
  }
}

function renderProjectList() {
  const list = byId("project-list");
  if (!app.projects.length) {
    list.innerHTML = '<div class="empty-state compact"><div class="empty-glyph" aria-hidden="true">◇</div><p>还没有项目</p></div>';
    return;
  }
  list.innerHTML = app.projects.map((project) => {
    const state = projectState(project);
    return `<button type="button" class="list-row ${project.id === app.selectedProject?.id ? "selected" : ""}" data-project-id="${escapeHtml(project.id)}" aria-label="查看项目 ${escapeHtml(project.workflow_id || project.id)}">
      <span class="row-main"><span class="row-title">${escapeHtml(project.workflow_id || project.id)}</span><span class="row-sub">${escapeHtml(project.step_count ?? 0)} 个步骤 · ${escapeHtml(shortId(project.id))}</span></span>
      <span class="status-pill ${state.tone}">${state.label}</span><span class="row-arrow" aria-hidden="true">›</span>
    </button>`;
  }).join("");
}

async function selectProject(id) {
  const detail = byId("project-detail");
  detail.innerHTML = '<div class="loading-line">正在读取输入文件...</div>';
  try {
    app.selectedProject = await api(`/api/projects/${encodeURIComponent(id)}`);
    app.selectedStep = app.selectedProject.steps?.[0]?.step_id || null;
    app.selectedFile = "INCAR";
    renderProjectList();
    renderProjectDetail();
  } catch (error) {
    detail.innerHTML = `<div class="notice bad">${escapeHtml(error.message)}</div>`;
  }
}

function selectedStep() {
  return app.selectedProject?.steps?.find((step) => step.step_id === app.selectedStep)
    || app.selectedProject?.steps?.[0];
}

function renderProjectDetail() {
  const element = byId("project-detail");
  const project = app.selectedProject;
  if (!project) {
    element.innerHTML = '<div class="detail-empty"><div class="empty-glyph" aria-hidden="true">◇</div><h2>选择一个项目</h2><p>输入文件和审核记录会显示在这里。</p></div>';
    return;
  }
  const card = project.review_card || {};
  const state = projectState(card);
  const step = selectedStep();
  const cardStep = card.steps?.find((item) => item.step_id === step?.step_id) || {};
  const files = ["POSCAR", "INCAR", "KPOINTS", "POTCAR 信息"];
  const activeFile = files.includes(app.selectedFile) ? app.selectedFile : "INCAR";
  const text = activeFile === "POTCAR 信息" ? "" : step?.inputs?.[activeFile] || "";
  const potentials = step?.potcar || {};
  const metadata = activeFile === "POTCAR 信息" ? `<div class="metadata">
    <div class="metadata-row"><span>赝势名称</span><strong>${escapeHtml((potentials.potential_names || cardStep.potcar_names || []).join(", ") || "—")}</strong></div>
    <div class="metadata-row"><span>SHA-256</span><strong class="hash">${escapeHtml(potentials.sha256 || cardStep.file_hashes?.POTCAR || "—")}</strong></div>
    <div class="metadata-row"><span>物种顺序</span><strong>${escapeHtml((cardStep.species_order || []).join(" · ") || "—")}</strong></div>
  </div>` : `<pre>${escapeHtml(text || "该文件不可用")}</pre>`;
  const review = card.status === "reviewed" ? `<div class="notice ${state.tone === "good" ? "good" : state.tone === "bad" ? "bad" : ""}">
      ${state.label} · 审核人 ${escapeHtml(card.reviewer || "—")}${card.notes ? `<br>${escapeHtml(card.notes)}` : ""}
    </div>` : `<form class="review-form" id="review-form">
      <label class="field"><span>审核人</span><input name="reviewer" required autocomplete="name" placeholder="姓名或集群用户名"></label>
      <div class="review-options" role="radiogroup" aria-label="审核决定">
        <label><input type="radio" name="decision" value="accept" required> 接受</label>
        <label><input type="radio" name="decision" value="revise"> 修改</label>
        <label><input type="radio" name="decision" value="reject"> 拒绝</label>
      </div>
      <label class="field"><span>审核备注</span><textarea name="notes" rows="3" placeholder="修改或拒绝时请说明原因"></textarea></label>
      <div class="review-actions"><button class="button button-primary" type="submit">提交审核</button></div>
      <div id="review-message" class="inline-message" role="status" aria-live="polite"></div>
    </form>`;
  element.innerHTML = `<div class="detail-heading"><div><h2>${escapeHtml(card.workflow_id || project.id)}</h2><p class="detail-kicker">${escapeHtml(project.id)}</p></div><span class="status-pill ${state.tone}">${state.label}</span></div>
    <div class="summary-grid">
      <div class="summary-item"><span>输入完整性</span><strong>${card.integrity_valid ? "已校验" : "校验失败"}</strong></div>
      <div class="summary-item"><span>计算步骤</span><strong>${escapeHtml(card.steps?.length ?? project.steps?.length ?? 0)}</strong></div>
      <div class="summary-item"><span>审核状态</span><strong>${state.label}</strong></div>
    </div>
    <div class="detail-section"><h3>输入文件</h3>
      <div class="tabs" role="tablist" aria-label="计算步骤">${(project.steps || []).map((item) => `<button type="button" role="tab" aria-selected="${item.step_id === step?.step_id}" class="tab ${item.step_id === step?.step_id ? "active" : ""}" data-step="${escapeHtml(item.step_id)}">${escapeHtml(item.step_id)}</button>`).join("")}</div>
      <div class="summary-grid">
        <div class="summary-item"><span>任务</span><strong>${escapeHtml(cardStep.task || "—")}</strong></div>
        <div class="summary-item"><span>ENCUT</span><strong>${escapeHtml(cardStep.encut ?? "—")} eV</strong></div>
        <div class="summary-item"><span>k 点</span><strong>${escapeHtml(cardStep.kpoints_summary || "—")}</strong></div>
        <div class="summary-item"><span>磁性模式</span><strong>${escapeHtml(cardStep.magnetic_mode || "—")}</strong></div>
        <div class="summary-item"><span>物种 / 原子数</span><strong>${escapeHtml((cardStep.species_order || []).map((name, i) => `${name} ${cardStep.counts?.[i] ?? ""}`).join(" · ") || "—")}</strong></div>
        <div class="summary-item"><span>MAGMOM</span><strong>${escapeHtml((cardStep.magmom || []).join(" ") || "—")}</strong></div>
      </div>
      <div class="tabs" role="tablist" aria-label="输入文件">${files.map((name) => `<button type="button" role="tab" aria-selected="${name === activeFile}" class="tab ${name === activeFile ? "active" : ""}" data-file="${escapeHtml(name)}">${escapeHtml(name)}</button>`).join("")}</div>
      <div class="file-panel"><div class="file-panel-head"><span>${escapeHtml(activeFile)}</span><span>${activeFile === "POTCAR 信息" ? "仅元数据" : "只读"}</span></div>${metadata}</div>
    </div>
    <div class="detail-section"><h3>输入审核</h3>${!card.integrity_valid ? '<div class="notice bad">输入完整性校验失败，无法审核。</div>' : review}</div>`;
}

async function submitProject(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = byId("project-submit");
  const body = new FormData(form);
  body.set("include_dos", form.elements.include_dos.checked ? "true" : "false");
  button.disabled = true;
  message("project-form-message", "正在生成输入文件...");
  try {
    const project = await api("/api/projects", { method: "POST", body });
    form.reset();
    message("project-form-message", "输入文件已生成，请检查并审核。", "success");
    await loadProjects(project.id);
    toast("输入文件已生成");
  } catch (error) {
    message("project-form-message", error.message, "error");
  } finally { button.disabled = !app.status?.project_generation_ready; }
}

async function submitReview(event) {
  event.preventDefault();
  const form = event.target;
  const data = new FormData(form);
  const decision = data.get("decision");
  const notes = String(data.get("notes") || "").trim();
  if (!decision) { message("review-message", "请选择审核决定。", "error"); return; }
  if (decision !== "accept" && !notes) { message("review-message", "修改或拒绝时需要填写审核备注。", "error"); return; }
  const button = form.querySelector('button[type="submit"]');
  button.disabled = true;
  message("review-message", "正在记录审核...");
  try {
    const project = await api(`/api/projects/${encodeURIComponent(app.selectedProject.id)}/review`, {
      method: "POST", body: { reviewer: String(data.get("reviewer") || "").trim(), decision, notes },
    });
    app.selectedProject = project;
    renderProjectDetail();
    await loadProjects(project.id);
    toast("审核已记录");
  } catch (error) {
    message("review-message", error.message, "error");
    button.disabled = false;
  }
}

function sessionState(session) {
  if (!session) return { label: "未开始", tone: "muted" };
  if (session.status === "running") return { label: "正在执行", tone: "warn" };
  if (session.status === "pending_approval") return { label: "等待审核", tone: "warn" };
  if (session.status === "completed") return { label: "已完成", tone: "good" };
  return { label: "执行失败", tone: "bad" };
}

function renderSession() {
  const session = app.session;
  const badge = byId("agent-run-status");
  const state = sessionState(session);
  badge.textContent = state.label;
  badge.className = `status-pill ${state.tone}`;
  const element = byId("agent-session");
  if (!session) {
    element.innerHTML = '<div class="empty-state compact"><div class="empty-glyph" aria-hidden="true">◎</div><p>暂无 Agent 任务</p></div>';
    return;
  }
  let content = `<div class="agent-session-meta"><span>任务 ${escapeHtml(shortId(session.id))}</span><span>·</span><span>${state.label}</span></div>`;
  if (session.status === "running") content += '<div class="notice">Agent 正在处理任务，状态会自动更新。</div>';
  if (session.status === "error") content += `<div class="agent-error">${escapeHtml(session.error || "任务未完成")}</div>`;
  if (session.status === "pending_approval") {
    const approvals = session.pending_approvals || [];
    content += `<form id="approval-form"><div class="notice">以下每项调用均需由具名审核人单独决定；批准后仍会进行运行时校验。</div><div class="approval-list">${approvals.map((item) => `<section class="approval-item" data-call-id="${escapeHtml(item.call_id)}">
      <div class="approval-head"><strong>${escapeHtml(toolNames[item.tool] || item.tool)}</strong><span class="status-pill ${item.risk_tier === "cluster_execute" ? "bad" : "warn"}">${item.risk_tier === "cluster_execute" ? "集群操作" : "资源操作"}</span></div>
      <p>${escapeHtml(item.reason || item.tool)} · ${escapeHtml(item.call_id)}</p>
      <div class="review-options" role="radiogroup" aria-label="${escapeHtml(item.call_id)} 审批决定">
        <label class="decision-choice"><input type="radio" name="decision-${escapeHtml(item.call_id)}" value="approve" required> 批准</label>
        <label class="decision-choice"><input type="radio" name="decision-${escapeHtml(item.call_id)}" value="reject"> 拒绝</label>
      </div>
      <details><summary>查看调用参数</summary><pre>${escapeHtml(JSON.stringify(item.arguments || {}, null, 2))}</pre></details>
    </section>`).join("")}</div>
      <div class="approval-submit"><label class="field"><span>审核人</span><input name="reviewer" required autocomplete="name" placeholder="姓名或集群用户名"></label></div>
      <div class="form-actions"><button type="submit" class="button button-primary">提交决定</button></div>
      <div id="approval-message" class="inline-message" role="status" aria-live="polite"></div></form>`;
  }
  const result = session.result;
  if (result?.llm_output) content += `<div class="agent-answer">${escapeHtml(result.llm_output)}</div>`;
  if (result?.transcript) {
    const transcript = result.transcript;
    const count = transcript.results?.length ?? 0;
    content += `<details class="agent-transcript"><summary>执行记录 · ${count} 项</summary><pre>${escapeHtml(JSON.stringify(transcript, null, 2))}</pre></details>`;
  }
  element.innerHTML = content;
}

async function loadSession(id, quiet = false) {
  try {
    app.session = await api(`/api/agent/sessions/${encodeURIComponent(id)}`);
    renderSession();
    if (app.session.status === "running") scheduleSessionPoll(id);
    else clearTimeout(app.sessionTimer);
  } catch (error) {
    clearTimeout(app.sessionTimer);
    if (!quiet) toast(error.message, "error");
    app.session = { id, status: "error", error: error.message };
    renderSession();
  }
}

function scheduleSessionPoll(id) {
  clearTimeout(app.sessionTimer);
  app.sessionTimer = setTimeout(() => loadSession(id, true), 2500);
}

async function submitAgent(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const goal = form.elements.goal.value.trim();
  if (!goal) return;
  const button = byId("agent-submit");
  button.disabled = true;
  message("agent-form-message", "正在创建任务...");
  try {
    const session = await api("/api/agent/sessions", { method: "POST", body: { goal } });
    app.session = session;
    try { localStorage.setItem("magvasp-session-id", session.id); } catch { /* private mode */ }
    renderSession();
    scheduleSessionPoll(session.id);
    message("agent-form-message", "");
    toast("Agent 任务已开始");
  } catch (error) {
    message("agent-form-message", error.message, "error");
  } finally { button.disabled = !app.status?.agent_configured; }
}

async function submitApproval(event) {
  event.preventDefault();
  const form = event.target;
  const reviewer = form.elements.reviewer.value.trim();
  const decisions = {};
  for (const item of app.session?.pending_approvals || []) {
    const selected = form.querySelector(`input[name="decision-${CSS.escape(item.call_id)}"]:checked`);
    if (!selected) { message("approval-message", "请对每项调用作出决定。", "error"); return; }
    decisions[item.call_id] = selected.value === "approve";
  }
  const button = form.querySelector('button[type="submit"]');
  button.disabled = true;
  message("approval-message", "正在提交决定...");
  try {
    app.session = await api(`/api/agent/sessions/${encodeURIComponent(app.session.id)}/decisions`, {
      method: "POST", body: { reviewer, decisions },
    });
    renderSession();
    if (app.session.status === "running") scheduleSessionPoll(app.session.id);
    toast("决定已记录");
  } catch (error) {
    message("approval-message", error.message, "error");
    button.disabled = false;
  }
}

async function loadResults(preferredId = app.selectedResult?.id) {
  const list = byId("result-list");
  list.innerHTML = '<div class="loading-line">正在读取结果...</div>';
  try {
    const data = await api("/api/results");
    app.results = data.results || [];
    byId("result-count").textContent = `${app.results.length} 份结果`;
    renderResultList();
    if (preferredId && app.results.some((item) => item.id === preferredId)) await selectResult(preferredId);
  } catch (error) {
    list.innerHTML = `<div class="notice bad">${escapeHtml(error.message)}</div>`;
  }
}

function renderResultList() {
  const list = byId("result-list");
  if (!app.results.length) {
    list.innerHTML = '<div class="empty-state compact"><div class="empty-glyph" aria-hidden="true">▤</div><p>还没有分析记录</p></div>';
    return;
  }
  list.innerHTML = app.results.map((result) => `<button type="button" class="list-row ${result.id === app.selectedResult?.id ? "selected" : ""}" data-result-id="${escapeHtml(result.id)}" aria-label="查看结果 ${escapeHtml(result.id)}">
    <span class="row-main"><span class="row-title">${escapeHtml(shortId(result.id))}</span><span class="row-sub">${result.final_energy == null ? "能量待解析" : `${escapeHtml(formatNumber(result.final_energy))} eV`}</span></span>
    <span class="status-pill ${result.converged ? "good" : "warn"}">${result.converged ? "已收敛" : "未确认收敛"}</span><span class="row-arrow" aria-hidden="true">›</span>
  </button>`).join("");
}

async function selectResult(id) {
  const element = byId("result-detail");
  element.innerHTML = '<div class="loading-line">正在读取分析结果...</div>';
  try {
    app.selectedResult = await api(`/api/results/${encodeURIComponent(id)}`);
    renderResultList();
    renderResultDetail();
  } catch (error) {
    element.innerHTML = `<div class="notice bad">${escapeHtml(error.message)}</div>`;
  }
}

function safePlotUrl(value) {
  const path = String(value || "");
  return /^\/api\/results\/[A-Za-z0-9_.-]+\/plots\/[A-Za-z0-9_.-]+$/.test(path) && !path.includes("..") ? path : null;
}

function formatMagnetization(summary) {
  if (summary.total_magnetization != null) {
    return `${formatNumber(summary.total_magnetization)} μB`;
  }
  const vector = summary.total_magnetization_vector;
  if (Array.isArray(vector) && vector.length === 3) {
    return `[${vector.map((item) => formatNumber(item, 4)).join(", ")}] μB`;
  }
  return "—";
}

function renderResultDetail() {
  const element = byId("result-detail");
  const result = app.selectedResult;
  if (!result) {
    element.innerHTML = '<div class="detail-empty"><div class="empty-glyph" aria-hidden="true">▤</div><h2>选择一份结果</h2><p>收敛信息与图像会显示在这里。</p></div>';
    return;
  }
  const summary = result.summary || {};
  const converged = summary.converged === true;
  const plots = Object.entries(result.plots || {}).filter(([, url]) => safePlotUrl(url));
  const plotProvenance = plots.length && result.plot_provenance === "unverified_external_tables"
    ? '<div class="notice">图像来自导入的数据表，尚未核对其与本次 VASP 输出的来源对应关系。</div>'
    : "";
  const warnings = summary.warnings || [];
  const moments = summary.ion_magnetizations || [];
  const figures = plots.length ? plots.map(([name, url]) => `<figure class="plot-frame"><figcaption>${escapeHtml(resultPlotNames[name] || name)}</figcaption><a href="${escapeHtml(url)}" target="_blank" rel="noopener" aria-label="打开${escapeHtml(resultPlotNames[name] || name)}图像"><img src="${escapeHtml(url)}" alt="${escapeHtml(resultPlotNames[name] || name)}图像" loading="lazy"></a></figure>`).join("") : '<div class="notice">这份结果没有可显示的图像。</div>';
  const momentRows = moments.length ? `<div class="moment-table"><div class="moment-table-head"><span>原子</span><span>元素</span><span>磁矩 (μB)</span></div>${moments.map((ion) => `<div class="moment-table-row"><span>${escapeHtml(ion.index ?? "—")}</span><span>${escapeHtml(ion.species || "—")}</span><span>${escapeHtml(formatNumber(ion.moment, 4))}</span></div>`).join("")}</div>` : '<span class="row-sub">无逐原子磁矩数据</span>';
  element.innerHTML = `<div class="detail-heading"><div><h2>计算结果 ${escapeHtml(shortId(result.id))}</h2><p class="detail-kicker">${escapeHtml(result.id)}</p></div><span class="status-pill ${converged ? "good" : "warn"}">${converged ? "已收敛" : "未确认收敛"}</span></div>
    <div class="metric-grid">
      <div class="metric"><span>总能量</span><strong>${escapeHtml(formatNumber(summary.final_energy))} eV</strong></div>
      <div class="metric"><span>费米能</span><strong>${escapeHtml(formatNumber(summary.fermi_energy))} eV</strong></div>
      <div class="metric"><span>总磁矩</span><strong>${escapeHtml(formatMagnetization(summary))}</strong></div>
    </div>
    ${!converged ? '<div class="notice">输出尚未确认科学收敛，请检查计算日志与解析警告。</div>' : ""}
    ${warnings.length ? `<div class="detail-section"><h3>解析警告</h3><ul class="warning-list">${warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join("")}</ul></div>` : ""}
    <div class="detail-section"><h3>图像</h3>${plotProvenance}${figures}</div>
    <div class="detail-section"><h3>逐原子磁矩</h3>${momentRows}</div>
    <div class="detail-section"><h3>来源</h3><div class="metadata-row"><span>结果 SHA-256</span><strong class="hash">${escapeHtml(summary.result_sha256 || "—")}</strong></div></div>`;
}

async function submitResult(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const body = new FormData(form);
  const button = byId("result-submit");
  button.disabled = true;
  message("result-form-message", "正在解析输出文件...");
  try {
    const result = await api("/api/results", { method: "POST", body });
    form.reset();
    message("result-form-message", "结果分析完成。", "success");
    await loadResults(result.id);
    toast("结果分析完成");
  } catch (error) {
    message("result-form-message", error.message, "error");
  } finally { button.disabled = false; }
}

function bindEvents() {
  window.addEventListener("hashchange", () => setView(currentView()));
  byId("menu-toggle").addEventListener("click", () => {
    const open = byId("sidebar").classList.toggle("open");
    byId("menu-toggle").setAttribute("aria-expanded", String(open));
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      byId("sidebar").classList.remove("open");
      byId("menu-toggle").setAttribute("aria-expanded", "false");
    }
  });
  byId("project-form").addEventListener("submit", submitProject);
  byId("project-detail").addEventListener("submit", (event) => {
    if (event.target.id === "review-form") submitReview(event);
  });
  byId("project-detail").addEventListener("click", (event) => {
    const step = event.target.closest("[data-step]");
    const file = event.target.closest("[data-file]");
    if (step) { app.selectedStep = step.dataset.step; renderProjectDetail(); }
    if (file) { app.selectedFile = file.dataset.file; renderProjectDetail(); }
  });
  byId("project-list").addEventListener("click", (event) => {
    const row = event.target.closest("[data-project-id]");
    if (row) selectProject(row.dataset.projectId);
  });
  byId("refresh-projects").addEventListener("click", () => loadProjects());
  byId("agent-form").addEventListener("submit", submitAgent);
  byId("agent-session").addEventListener("submit", (event) => {
    if (event.target.id === "approval-form") submitApproval(event);
  });
  byId("result-form").addEventListener("submit", submitResult);
  byId("result-list").addEventListener("click", (event) => {
    const row = event.target.closest("[data-result-id]");
    if (row) selectResult(row.dataset.resultId);
  });
  byId("refresh-results").addEventListener("click", () => loadResults());
  byId("refresh-status").addEventListener("click", () => loadStatus().catch((error) => toast(error.message, "error")));
}

async function start() {
  bindEvents();
  setView(currentView());
  try { await loadStatus(); } catch { /* each view retains its own connection state */ }
  await Promise.all([loadProjects(), loadResults()]);
  let sessionId;
  try { sessionId = localStorage.getItem("magvasp-session-id"); } catch { sessionId = null; }
  if (sessionId) loadSession(sessionId, true);
}

start();
