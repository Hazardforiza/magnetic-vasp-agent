"use strict";

const app = {
  csrfToken: "",
  status: null,
  nodes: [],
  nodeProfile: null,
  selectedNode: null,
  nodeProbeToken: 0,
  resourceDrafts: {},
  resourceObservations: {},
  resourceProbes: new Set(),
  resourceBusy: null,
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
  nodes: "节点",
  results: "结果",
  knowledge: "知识库",
  settings: "环境",
};

const toolNames = {
  "knowledge.library_search": "检索知识库",
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
  if (view === "knowledge" && typeof knowledgeEntered === "function") knowledgeEntered();
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
      settingRow("集群节点 profile", status.execution_profile_configured, "已配置", "未配置"),
    ].join("");
    byId("service-origin").textContent = location.origin;
    if (!status.agent_configured) message("agent-form-message", "模型配置尚未就绪。", "error");
    else message("agent-form-message", "");
    if (!status.project_generation_ready) message("project-form-message", "输入生成环境尚未就绪。", "error");
    else message("project-form-message", "");
    refreshResourceControls();
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

function nodeState(node) {
  const raw = String(node?.status || node?.state || "").toLowerCase();
  const labels = {
    available: ["可连接", "good"], ready: ["可连接", "good"], ok: ["可连接", "good"],
    observable: ["可探测", "good"], connection_failed: ["连接失败", "bad"],
    idle: ["观察到空闲", "good"], busy: ["有占用", "warn"], occupied: ["有占用", "warn"],
    checking: ["检查中", "warn"], running: ["检查中", "warn"], stale: ["已过期", "warn"],
    unknown: ["未检查", "muted"], unconfigured: ["未配置", "muted"],
    unavailable: ["连接失败", "bad"], unreachable: ["连接失败", "bad"], error: ["检查失败", "bad"],
    failed: ["检查失败", "bad"],
  };
  const [defaultLabel, tone] = labels[raw] || [raw ? raw : "未检查", raw ? "warn" : "muted"];
  return { label: node?.status_label || node?.label || defaultLabel, tone };
}

function nodeName(node) {
  return node?.name || node?.node || node?.requested_node || "";
}

function nodeSnapshot(node) {
  if (!node) return {};
  const nested = node.snapshot || node.summary || {};
  return { ...nested, ...node };
}

function nodeMemoryGiB(node) {
  const value = node?.mem_available_kib ?? node?.memory_available_kib;
  const number = Number(value);
  return Number.isFinite(number) ? `${formatNumber(number / (1024 * 1024), 1)} GiB` : "—";
}

function nodeCpuCount(node) {
  if (Array.isArray(node?.cpus)) return node.cpus.length;
  return node?.cpu_count ?? node?.logical_cpus ?? node?.cpus_total ?? "—";
}

function nodeAvailability(node) {
  const processes = node?.process_count ?? (Array.isArray(node?.processes) ? node.processes.length : null);
  const cpu = node?.cpu_percent ?? node?.active_cpu_percent ?? node?.total_cpu_percent;
  const parts = [];
  if (processes != null) parts.push(`${processes} 个进程`);
  if (cpu != null && Number.isFinite(Number(cpu))) parts.push(`CPU ${formatNumber(cpu, 1)}%`);
  return parts.join(" · ") || "占用情况待探测";
}

function renderNodeList() {
  const list = byId("node-list");
  if (!app.nodes.length) {
    const configured = app.nodeProfile?.configured;
    list.innerHTML = configured === false
      ? '<div class="notice">尚未配置集群节点。请先准备本机的执行配置。</div>'
      : '<div class="empty-state compact"><div class="empty-glyph" aria-hidden="true">◌</div><p>暂无可选择的节点</p></div>';
    return;
  }
  list.innerHTML = app.nodes.map((item) => {
    const node = nodeSnapshot(item);
    const state = nodeState(node);
    const name = nodeName(item) || nodeName(node) || "—";
    const observed = ["observable", "busy"].includes(node.status) && node.observed_at;
    const sub = [node.hostname || (observed ? "已探测" : "未探测"), nodeCpuCount(node) === "—" ? null : `${nodeCpuCount(node)} CPU`].filter(Boolean).join(" · ");
    return `<button type="button" class="list-row ${name === nodeName(app.selectedNode) ? "selected" : ""}" data-node-id="${escapeHtml(name)}" aria-label="查看节点 ${escapeHtml(name)}">
      <span class="row-main"><span class="row-title">${escapeHtml(name)}</span><span class="row-sub">${escapeHtml(sub || nodeAvailability(node))}</span></span>
      <span class="status-pill ${state.tone}">${escapeHtml(state.label)}</span><span class="row-arrow" aria-hidden="true">›</span>
    </button>`;
  }).join("");
}

function renderNodeDetail() {
  const element = byId("node-detail");
  const node = nodeSnapshot(app.selectedNode);
  if (!app.selectedNode) {
    element.innerHTML = '<div class="detail-empty"><div class="empty-glyph" aria-hidden="true">◌</div><h2>选择一个节点</h2><p>节点负载、运行时和 Wannier 能力会显示在这里。</p></div>';
    return;
  }
  const name = nodeName(app.selectedNode) || nodeName(node) || "—";
  const state = nodeState(node);
  const vasp = node.vasp || node.runtime || {};
  const wannier = node.wannier || node.wannier90 || {};
  const moduleLoaded = node.module_loaded ?? vasp.module_loaded;
  const executable = node.executables || vasp.executables || {};
  const w90Capability = wannier.capability_verified ?? wannier.supported ?? node.wannier_capability_verified;
  const w90Version = wannier.wannier90_version ?? wannier.version ?? node.wannier90_version;
  const w90Path = wannier.wannier90_path ?? node.wannier90_path;
  const vaspNclPath = wannier.vasp_ncl_path ?? node.vasp_ncl_path;
  const rawError = node.error || node.probe_error || node.detail;
  const error = rawError && typeof rawError === "object"
    ? (rawError.message || rawError.code || JSON.stringify(rawError)) : rawError;
  const observed = node.observed_at || node.checked_at || "—";
  const connectionNote = ["observable", "busy"].includes(node.status)
    ? "本次节点探测已完成" : "尚未完成节点探测";
  const probeButton = `<button type="button" class="button button-small" data-node-probe="${escapeHtml(name)}">重新检查</button>`;
  element.innerHTML = `<div class="detail-heading"><div><h2 id="node-detail-title">${escapeHtml(name)}</h2><p class="detail-kicker">${escapeHtml(node.hostname || connectionNote)}</p></div><div class="review-actions"><span class="status-pill ${state.tone}">${escapeHtml(state.label)}</span>${probeButton}</div></div>
    ${error ? `<div class="notice bad">${escapeHtml(error)}</div>` : ""}
    <div class="metric-grid">
      <div class="metric"><span>逻辑 CPU</span><strong>${escapeHtml(nodeCpuCount(node))}</strong></div>
      <div class="metric"><span>负载 (1 min)</span><strong>${escapeHtml(formatNumber(node.load1, 2))}</strong></div>
      <div class="metric"><span>可用内存</span><strong>${escapeHtml(nodeMemoryGiB(node))}</strong></div>
    </div>
    <div class="detail-section"><h3>只读观察</h3><div class="metadata">
      <div class="metadata-row"><span>最近探测</span><strong>${escapeHtml(observed)}</strong></div>
      <div class="metadata-row"><span>状态说明</span><strong>${escapeHtml(node.status_reason || "—")}</strong></div>
      <div class="metadata-row"><span>占用概况</span><strong>${escapeHtml(nodeAvailability(node))}</strong></div>
      <div class="metadata-row"><span>可访问 CPU</span><strong>${escapeHtml(Array.isArray(node.allowed_cpu_ids) ? node.allowed_cpu_ids.length : node.accessible_cpu_count ?? "—")}</strong></div>
    </div></div>
    <div class="detail-section"><h3>VASP 运行时</h3><div class="metadata">
      <div class="metadata-row"><span>模块状态</span><strong>${moduleLoaded == null ? "未检查" : moduleLoaded ? "已加载" : "未加载"}</strong></div>
      <div class="metadata-row"><span>vasp_std</span><strong>${executable.vasp_std == null ? "未检查" : executable.vasp_std ? "可用" : "不可用"}</strong></div>
      <div class="metadata-row"><span>vasp_ncl</span><strong>${executable.vasp_ncl == null ? (vaspNclPath ? "可用" : "未检查") : executable.vasp_ncl ? "可用" : "不可用"}</strong></div>
      <div class="metadata-row"><span>vasp_gam</span><strong>${executable.vasp_gam == null ? "未检查" : executable.vasp_gam ? "可用" : "不可用"}</strong></div>
    </div></div>
    <div class="detail-section"><h3>Wannier90 能力</h3><div class="metadata">
      <div class="metadata-row"><span>能力状态</span><strong>${w90Capability == null ? "未检查" : w90Capability ? "已验证" : "未验证"}</strong></div>
      <div class="metadata-row"><span>版本</span><strong>${escapeHtml(w90Version || "—")}</strong></div>
      <div class="metadata-row"><span>wannier90.x</span><strong class="hash">${escapeHtml(w90Path || "—")}</strong></div>
      <div class="metadata-row"><span>VASP NCL</span><strong class="hash">${escapeHtml(vaspNclPath || "—")}</strong></div>
    </div></div>`;
}

async function loadNodes(preferredName = nodeName(app.selectedNode)) {
  const list = byId("node-list");
  list.innerHTML = '<div class="loading-line">正在读取节点...</div>';
  try {
    const data = await api("/api/nodes");
    app.nodeProfile = data;
    app.nodes = Array.isArray(data.nodes) ? data.nodes : [];
    byId("node-count").textContent = `${app.nodes.length} 个节点`;
    renderNodeList();
    refreshResourceControls();
    if (preferredName && app.nodes.some((item) => nodeName(item) === preferredName)) {
      app.selectedNode = app.nodes.find((item) => nodeName(item) === preferredName);
      renderNodeList();
      renderNodeDetail();
    } else if (app.selectedNode) {
      app.selectedNode = null;
      renderNodeDetail();
    }
  } catch (error) {
    app.nodes = [];
    byId("node-count").textContent = "";
    list.innerHTML = `<div class="notice bad">${escapeHtml(error.message)}</div>`;
    app.selectedNode = null;
    renderNodeDetail();
    refreshResourceControls();
  }
}

async function selectNode(name, probe = true) {
  const base = app.nodes.find((item) => nodeName(item) === name);
  if (!base) return;
  app.selectedNode = probe ? { ...base, name, status: "checking" } : base;
  if (probe) app.nodes = app.nodes.map((item) => nodeName(item) === name ? app.selectedNode : item);
  renderNodeList();
  renderNodeDetail();
  if (!probe) return;
  const token = ++app.nodeProbeToken;
  byId("node-detail").innerHTML = '<div class="loading-line">正在读取节点状态...</div>';
  try {
    const data = await api(`/api/nodes/${encodeURIComponent(name)}/probe`);
    if (token !== app.nodeProbeToken) return;
    const merged = { ...base, ...data, name: data.name || data.node || name };
    app.resourceObservations[name] = merged;
    app.selectedNode = merged;
    app.nodes = app.nodes.map((item) => nodeName(item) === name ? merged : item);
    renderNodeList();
    renderNodeDetail();
    refreshResourceControls();
  } catch (error) {
    if (token !== app.nodeProbeToken) return;
    app.selectedNode = { ...base, name, status: "error", error: error.message };
    app.nodes = app.nodes.map((item) => nodeName(item) === name ? app.selectedNode : item);
    renderNodeList();
    renderNodeDetail();
    refreshResourceControls();
  }
}

function resourcePreferences(project = app.selectedProject) {
  const resources = project?.resources || {};
  return {
    mode: resources.mode === "manual" ? "manual" : "llm",
    node: resources.node || "",
    mpi_ranks: resources.mpi_ranks ?? 1,
  };
}

function resourceFormValues(form) {
  const manual = form.elements.resource_mode.value === "manual";
  return {
    mode: manual ? "manual" : "llm",
    node: manual ? form.elements.resource_node.value : null,
    mpi_ranks: manual ? Number(form.elements.resource_mpi_ranks.value) : null,
  };
}

function resourceDraft(project = app.selectedProject) {
  return app.resourceDrafts[project.id] || resourcePreferences(project);
}

function syncResourceNodeOptions(form) {
  const select = form.elements.resource_node;
  const value = select.value || select.dataset.savedNode || "";
  const names = app.nodes.map(nodeName).filter((name) => name && name !== "mu01");
  const options = ['<option value="">请选择节点</option>'];
  if (value && !names.includes(value)) options.push(`<option value="${escapeHtml(value)}" disabled>${escapeHtml(value)}（当前未列出）</option>`);
  options.push(...names.map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`));
  select.innerHTML = options.join("");
  select.value = value;
  select.dataset.savedNode = value;
}

function freshResourceCoreCount(node) {
  const observed = Date.parse(node?.observed_at || "");
  const count = Number(node?.snapshot?.accessible_physical_core_count);
  return Number.isFinite(observed) && Date.now() - observed >= 0 && Date.now() - observed < 60000
    && Number.isInteger(count) && count > 0 ? count : null;
}

function updateResourceControls(form) {
  if (!form?.elements.resource_mode) return;
  const manual = form.elements.resource_mode.value === "manual";
  const configured = app.nodeProfile?.configured === true;
  const busy = form.id === "project-resources-form" && app.resourceBusy === app.selectedProject?.id;
  const node = form.elements.resource_node.value;
  const observation = app.resourceObservations[node];
  const cores = freshResourceCoreCount(observation);
  const manualOption = form.elements.resource_mode.querySelector('option[value="manual"]');
  manualOption.disabled = !configured;
  form.elements.resource_mode.disabled = busy;
  form.querySelector("[data-resource-manual]").hidden = !manual;
  form.elements.resource_node.disabled = busy || !manual || !configured;
  form.elements.resource_node.required = manual;
  form.elements.resource_mpi_ranks.disabled = busy || !manual || !configured;
  form.elements.resource_mpi_ranks.required = manual;
  form.elements.resource_mpi_ranks.max = String(Math.min(cores || 1024, 1024));
  const probe = form.querySelector("[data-resource-probe]");
  probe.disabled = busy || !manual || !configured || !node || app.resourceProbes.has(node);
  probe.textContent = app.resourceProbes.has(node) ? "检查中…" : "检查节点状态";
  const preview = form.querySelector("[data-resource-preview]");
  if (observation) {
    const snapshot = nodeSnapshot(observation);
    const state = nodeState(observation);
    const error = observation.error?.message || observation.error;
    preview.innerHTML = `<div class="resource-observation"><span class="status-pill ${state.tone}">${escapeHtml(state.label)}</span>
      ${error ? `<span>${escapeHtml(error)}</span>` : `<span>可访问物理核 ${escapeHtml(snapshot.accessible_physical_core_count ?? "—")}</span><span>负载 ${escapeHtml(formatNumber(snapshot.load1, 2))}</span><span>可用内存 ${escapeHtml(nodeMemoryGiB(snapshot))}</span>`}
      <small>${escapeHtml(observation.observed_at || "尚无有效观察")}${cores ? "" : " · 使用前需重新检查核数"}</small></div>`;
  } else preview.innerHTML = node ? '<span class="row-sub">尚未检查此节点，点击按钮读取当前状态。</span>' : "";
  const hint = form.querySelector("[data-resource-hint]");
  hint.textContent = !configured
    ? "尚未配置集群连接。可先保存由 LLM 选择，完成配置后再检查资源。"
    : manual
      ? `核数按物理核计，每核运行 1 个 MPI 进程。${cores ? `本次观察最多可访问 ${cores} 个物理核。` : "检查节点后可查看当前核数与负载。"}`
      : "LLM 会根据节点状态与项目规模选择节点和核数。";
  const selectButton = form.querySelector("[data-resource-select]");
  if (selectButton) {
    selectButton.textContent = busy ? "正在检查与选择…" : manual ? "检查所选资源" : "让 LLM 选择";
    selectButton.disabled = busy || !configured || (!manual && !app.status?.agent_configured);
    const saveButton = form.querySelector('[type="submit"]');
    saveButton.disabled = busy || (manual && !configured);
    const planButton = form.querySelector("[data-project-agent]");
    if (planButton) planButton.disabled = busy || !app.status?.agent_configured || resourceFormChanged(form) || !app.selectedProject?.resources?.selection;
    if (!manual && configured && !app.status?.agent_configured) hint.textContent = "模型尚未配置。可以先保存偏好，完成模型配置后再让 LLM 选择。";
  }
}

function refreshResourceControls() {
  for (const form of [byId("project-form"), byId("project-resources-form")]) {
    if (!form) continue;
    syncResourceNodeOptions(form);
    updateResourceControls(form);
  }
}

function resourceFormChanged(form) {
  const current = resourceFormValues(form);
  const stored = resourcePreferences();
  return current.mode !== stored.mode || (current.mode === "manual"
    && (current.node !== stored.node || current.mpi_ranks !== Number(stored.mpi_ranks)));
}

function resourceFormEdited(form) {
  form.elements.resource_node.dataset.savedNode = form.elements.resource_node.value;
  if (form.id === "project-resources-form" && app.selectedProject) {
    app.resourceDrafts[app.selectedProject.id] = {
      mode: form.elements.resource_mode.value,
      node: form.elements.resource_node.value,
      mpi_ranks: form.elements.resource_mpi_ranks.value,
    };
    message("resource-message", resourceFormChanged(form) ? "选择已修改，保存后用于此项目。" : "");
  }
  updateResourceControls(form);
}

async function probeResourceNode(form) {
  const name = form.elements.resource_node.value;
  if (!name || !app.nodes.some((item) => nodeName(item) === name) || app.resourceProbes.has(name)) return;
  app.resourceProbes.add(name);
  refreshResourceControls();
  try {
    const data = await api(`/api/nodes/${encodeURIComponent(name)}/probe`);
    app.resourceObservations[name] = data;
    app.nodes = app.nodes.map((item) => nodeName(item) === name ? { ...item, ...data } : item);
    renderNodeList();
  } catch (error) {
    app.resourceObservations[name] = { name, status: "error", error: { message: error.message } };
  } finally {
    app.resourceProbes.delete(name);
    refreshResourceControls();
  }
}

function renderResourceEditor(project) {
  const preferences = resourceDraft(project);
  const selection = project.resources?.selection;
  const blockers = selection?.assessment?.blockers || [];
  const selectionMarkup = selection ? `<div class="resource-selection"><div class="resource-selection-heading"><strong>${escapeHtml(selection.node)} · ${escapeHtml(selection.mpi_ranks)} 个物理核</strong><span class="status-pill good">${selection.source === "llm" ? "LLM 建议" : "已检查"}</span></div>
    <p>${escapeHtml(selection.reason || "资源方案已保存。")}</p><small>规划步骤：${escapeHtml(selection.execution_request?.step_id || "—")} · 观察时间：${escapeHtml(selection.observed_at || "—")}</small>
    ${blockers.length ? `<div class="notice">${blockers.map((blocker) => escapeHtml(blocker.code === "ALLOCATION_UNCONFIRMED" ? "仍需具名授权使用所选资源；节点状态只代表本次观察。" : typeof blocker === "string" ? blocker : blocker.message || blocker.code)).join("<br>")}</div>` : ""}</div>`
    : '<div class="notice">尚未确定节点与核数。可检查手动选择，或让 LLM 根据节点状态提出建议。</div>';
  return `<div class="detail-section"><h3>计算资源</h3><form id="project-resources-form" class="resource-form">
    <label class="field"><span>节点与核数</span><select name="resource_mode"><option value="llm" ${preferences.mode === "llm" ? "selected" : ""}>由 LLM 选择</option><option value="manual" ${preferences.mode === "manual" ? "selected" : ""}>手动指定</option></select></label>
    <div class="form-grid resource-manual" data-resource-manual ${preferences.mode === "manual" ? "" : "hidden"}>
      <label class="field"><span>提交节点</span><select name="resource_node" aria-label="项目提交节点" data-saved-node="${escapeHtml(preferences.node)}"><option value="${escapeHtml(preferences.node)}">${escapeHtml(preferences.node || "请选择节点")}</option></select></label>
      <label class="field"><span>物理核数</span><input name="resource_mpi_ranks" type="number" min="1" max="1024" step="1" value="${escapeHtml(preferences.mpi_ranks)}"><small>每个物理核运行 1 个 MPI 进程</small></label>
      <div class="resource-probe-row field-wide"><button type="button" class="button button-small" data-resource-probe>检查节点状态</button></div>
      <div class="resource-node-preview field-wide" data-resource-preview aria-live="polite"></div>
    </div>
    <p class="resource-hint" data-resource-hint></p>
    <div class="resource-actions"><button type="submit" class="button button-small">保存资源偏好</button><button type="button" class="button button-primary button-small" data-resource-select>让 LLM 选择</button><button type="button" class="button button-small" data-project-agent>交给 Agent 规划</button></div>
    <div id="resource-message" class="inline-message" role="status" aria-live="polite"></div>
    ${selectionMarkup}<p class="resource-hint">资源选择用于后续执行规划，实际计算仍需完成输入审核与资源授权。</p>
  </form></div>`;
}

async function saveProjectResources(event, choose = false) {
  event?.preventDefault();
  const form = byId("project-resources-form");
  const id = app.selectedProject?.id;
  if (!id || app.resourceBusy || !form.reportValidity()) return;
  const values = resourceFormValues(form);
  const changed = resourceFormChanged(form);
  app.resourceBusy = id;
  updateResourceControls(form);
  message("resource-message", choose ? (values.mode === "manual" ? "正在检查所选节点与核数…" : "正在读取节点状态并请求 LLM 选择…") : "正在保存资源偏好…");
  try {
    let project = app.selectedProject;
    if (changed) {
      project = await api(`/api/projects/${encodeURIComponent(id)}/resources`, { method: "PUT", body: values });
      if (app.selectedProject?.id === id) {
        app.selectedProject = project;
        renderProjectDetail();
        message("resource-message", choose ? "资源偏好已保存，正在检查与选择…" : "资源偏好已保存。");
      }
    }
    if (choose) {
      if (app.selectedProject?.id === id) app.selectedProject = project;
      project = await api(`/api/projects/${encodeURIComponent(id)}/resources/select`, { method: "POST" });
    }
    delete app.resourceDrafts[id];
    if (app.selectedProject?.id === id) {
      app.selectedProject = project;
      renderProjectDetail();
      message("resource-message", choose ? "资源方案已保存，可交给 Agent 继续规划。" : "资源偏好已保存。", "success");
    }
    toast(choose ? "资源方案已保存" : "资源偏好已保存");
  } catch (error) {
    if (app.selectedProject?.id === id) message("resource-message", error.message, "error");
    else toast(error.message, "error");
  } finally {
    app.resourceBusy = null;
    const current = byId("project-resources-form");
    if (current) updateResourceControls(current);
  }
}

async function planProjectWithAgent() {
  const project = app.selectedProject;
  const form = byId("project-resources-form");
  if (!project?.resources?.selection || resourceFormChanged(form) || app.resourceBusy) return;
  app.resourceBusy = project.id;
  updateResourceControls(form);
  message("resource-message", "正在创建项目执行规划…");
  try {
    const session = await api("/api/agent/sessions", { method: "POST", body: {
      goal: "根据该项目的输入与已选节点、核数生成执行计划，报告仍需完成的审核。", project_id: project.id,
    } });
    app.session = session;
    try { localStorage.setItem("magvasp-session-id", session.id); } catch { /* private mode */ }
    renderSession();
    scheduleSessionPoll(session.id);
    location.hash = "agent";
    setView("agent");
    toast("项目已交给 Agent 规划");
  } catch (error) {
    if (app.selectedProject?.id === project.id) message("resource-message", error.message, "error");
  } finally {
    app.resourceBusy = null;
    const current = byId("project-resources-form");
    if (current) updateResourceControls(current);
  }
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
    ${renderResourceEditor(project)}
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
  refreshResourceControls();
}

async function submitProject(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = byId("project-submit");
  const body = new FormData(form);
  body.set("include_dos", form.elements.include_dos.checked ? "true" : "false");
  if (form.elements.resource_mode.value !== "manual") {
    body.delete("resource_node");
    body.delete("resource_mpi_ranks");
  }
  button.disabled = true;
  message("project-form-message", "正在生成输入文件...");
  try {
    const project = await api("/api/projects", { method: "POST", body });
    form.reset();
    form.elements.resource_node.dataset.savedNode = "";
    form.elements.resource_node.value = "";
    refreshResourceControls();
    message("project-form-message", "输入文件已生成，请检查并审核。", "success");
    await loadProjects(project.id);
    toast("输入文件已生成");
  } catch (error) {
    message("project-form-message", error.message, "error");
  } finally { button.disabled = !app.status?.project_generation_ready; }
}

function updateProjectTaskControls() {
  const form = byId("project-form");
  if (!form) return;
  const task = form.elements.task.value;
  const dos = byId("project-dos-option");
  const checkbox = form.elements.include_dos;
  const hint = byId("project-task-hint");
  const advanced = {
    magnetic_order: "从 mCIF 磁矩生成有界的共线磁序候选，并分别生成静态自洽步骤。",
    convergence: "生成独立的 ENCUT 和 k 点间距扫描；每个点都需要单独完成收敛检查。",
    mae: "生成 [001]、[100]、[110] 三个 SOC 自洽方向；仅接受共线 mCIF，结果需比较到收敛精度。",
  };
  const isStatic = task === "static";
  dos.hidden = !isStatic;
  checkbox.disabled = !isStatic;
  if (!isStatic) checkbox.checked = false;
  hint.textContent = advanced[task] || "静态自洽可选择生成依赖 CHGCAR 的 DOS 后续步骤。";
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
  byId("project-form").addEventListener("change", (event) => {
    if (event.target.name?.startsWith("resource_")) resourceFormEdited(event.currentTarget);
    if (event.target.name === "task") updateProjectTaskControls();
  });
  byId("project-form").addEventListener("click", (event) => {
    if (event.target.closest("[data-resource-probe]")) probeResourceNode(event.currentTarget);
  });
  byId("project-detail").addEventListener("submit", (event) => {
    if (event.target.id === "review-form") submitReview(event);
    if (event.target.id === "project-resources-form") saveProjectResources(event);
  });
  byId("project-detail").addEventListener("change", (event) => {
    if (event.target.name?.startsWith("resource_")) resourceFormEdited(event.target.closest("form"));
  });
  byId("project-detail").addEventListener("input", (event) => {
    if (event.target.name === "resource_mpi_ranks") resourceFormEdited(event.target.closest("form"));
  });
  byId("project-detail").addEventListener("click", (event) => {
    const step = event.target.closest("[data-step]");
    const file = event.target.closest("[data-file]");
    if (step) { app.selectedStep = step.dataset.step; renderProjectDetail(); }
    if (file) { app.selectedFile = file.dataset.file; renderProjectDetail(); }
    if (event.target.closest("[data-resource-probe]")) probeResourceNode(event.target.closest("form"));
    if (event.target.closest("[data-resource-select]")) saveProjectResources(event, true);
    if (event.target.closest("[data-project-agent]")) planProjectWithAgent();
  });
  byId("project-list").addEventListener("click", (event) => {
    const row = event.target.closest("[data-project-id]");
    if (row) selectProject(row.dataset.projectId);
  });
  byId("refresh-projects").addEventListener("click", () => loadProjects());
  byId("refresh-nodes").addEventListener("click", () => {
    const selectedName = nodeName(app.selectedNode);
    if (selectedName) selectNode(selectedName);
    else loadNodes();
  });
  byId("node-list").addEventListener("click", (event) => {
    const row = event.target.closest("[data-node-id]");
    if (row) selectNode(row.dataset.nodeId);
  });
  byId("node-detail").addEventListener("click", (event) => {
    const button = event.target.closest("[data-node-probe]");
    if (button) selectNode(button.dataset.nodeProbe);
  });
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
  updateProjectTaskControls();
}

async function start() {
  bindEvents();
  knowledgeInit();
  setView(currentView());
  try { await loadStatus(); } catch { /* each view retains its own connection state */ }
  await Promise.all([loadProjects(), loadNodes(), loadResults(), loadKnowledge()]);
  knowledgeRefreshSeeds();
  let sessionId;
  try { sessionId = localStorage.getItem("magvasp-session-id"); } catch { sessionId = null; }
  if (sessionId) loadSession(sessionId, true);
}

start();
