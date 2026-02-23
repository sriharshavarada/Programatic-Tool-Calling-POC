const els = {
  modelInput: document.getElementById("modelInput"),
  runLocalBtn: document.getElementById("runLocalBtn"),
  runSandboxBtn: document.getElementById("runSandboxBtn"),
  runTraditionalBtn: document.getElementById("runTraditionalBtn"),
  runAllBtn: document.getElementById("runAllBtn"),
  clearLogsBtn: document.getElementById("clearLogsBtn"),
  statusPill: document.getElementById("statusPill"),
  metricMode: document.getElementById("metricMode"),
  metricDuration: document.getElementById("metricDuration"),
  metricModel: document.getElementById("metricModel"),
  metricTokens: document.getElementById("metricTokens"),
  metricContainer: document.getElementById("metricContainer"),
  tokenUsage: document.getElementById("tokenUsage"),
  metadataBox: document.getElementById("metadataBox"),
  generatedCode: document.getElementById("generatedCode"),
  finalText: document.getElementById("finalText"),
  logStream: document.getElementById("logStream"),
  modalOverlay: document.getElementById("modalOverlay"),
  modalCloseBtn: document.getElementById("modalCloseBtn"),
  modalTitle: document.getElementById("modalTitle"),
  modalContent: document.getElementById("modalContent"),
  modalShell: document.querySelector(".modal-shell"),
};

const laneEls = {
  local_langgraph: {
    status: document.getElementById("lane-local-status"),
    duration: document.getElementById("lane-local-duration"),
    tokens: document.getElementById("lane-local-tokens"),
    promptPreview: document.getElementById("lane-local-prompt-preview"),
    prompt: document.getElementById("lane-local-prompt"),
    snippet: document.getElementById("lane-local-snippet"),
    code: document.getElementById("lane-local-code"),
    final: document.getElementById("lane-local-final"),
    logs: document.getElementById("lane-local-logs"),
    transcript: document.getElementById("lane-local-transcript"),
  },
  sandbox_langgraph: {
    status: document.getElementById("lane-sandbox-status"),
    duration: document.getElementById("lane-sandbox-duration"),
    tokens: document.getElementById("lane-sandbox-tokens"),
    promptPreview: document.getElementById("lane-sandbox-prompt-preview"),
    prompt: document.getElementById("lane-sandbox-prompt"),
    snippet: document.getElementById("lane-sandbox-snippet"),
    code: document.getElementById("lane-sandbox-code"),
    final: document.getElementById("lane-sandbox-final"),
    logs: document.getElementById("lane-sandbox-logs"),
    transcript: document.getElementById("lane-sandbox-transcript"),
  },
  traditional: {
    status: document.getElementById("lane-traditional-status"),
    duration: document.getElementById("lane-traditional-duration"),
    tokens: document.getElementById("lane-traditional-tokens"),
    promptPreview: document.getElementById("lane-traditional-prompt-preview"),
    prompt: document.getElementById("lane-traditional-prompt"),
    snippet: document.getElementById("lane-traditional-snippet"),
    code: document.getElementById("lane-traditional-code"),
    final: document.getElementById("lane-traditional-final"),
    logs: document.getElementById("lane-traditional-logs"),
    transcript: document.getElementById("lane-traditional-transcript"),
  },
};

function setText(el, value) {
  if (!el) return;
  el.textContent = value;
}

function sectionContentEl(detailsEl) {
  if (!detailsEl) return null;
  return detailsEl.querySelector(".code-box, .log-stream");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function laneModeClass(mode) {
  if (mode === "local_langgraph") return "local";
  if (mode === "sandbox_langgraph") return "sandbox";
  if (mode === "traditional") return "traditional";
  return "";
}

function classifyModalLine(line) {
  const classes = ["modal-line"];
  if (line.includes("[LOCAL]")) classes.push("local-tag");
  if (line.includes("[SANDBOX]")) classes.push("sandbox-tag");
  if (line.includes("[TRADITIONAL]")) classes.push("traditional-tag");
  if (line.includes("[TX]")) classes.push("tx-tag");
  if (line.includes("[RX]")) classes.push("rx-tag");
  if (
    line.includes("tool_result") ||
    line.includes("tool_use") ||
    line.includes("server_tool_use") ||
    line.includes("TOOL CALLED:")
  ) {
    classes.push("tool-tag");
  }
  if (/error|traceback|exception/i.test(line)) classes.push("error-tag");
  if (/^\s*[\{\}\[\],"]?\s*$/.test(line)) classes.push("dim-tag");
  return classes.join(" ");
}

function renderModalContent(content) {
  if (!els.modalContent) return;
  const lines = String(content || "").split("\n");
  let currentFlow = "flow-neutral";

  const html = lines.map((line) => {
    const trimmed = line.trim();
    const isHeader = /^\[(LOCAL|SANDBOX|TRADITIONAL)\]\[(TX|RX)\]/.test(trimmed);

    if (isHeader) {
      currentFlow = trimmed.includes("[TX]") ? "flow-tx" : "flow-rx";
    } else if (/tool_result|tool_use|server_tool_use/i.test(trimmed)) {
      // Highlight tool event lines, but don't lose current tx/rx context for following lines.
      currentFlow = currentFlow || "flow-neutral";
    }

    let flowClass = currentFlow || "flow-neutral";
    if (/tool_result|tool_use|server_tool_use|TOOL CALLED:/i.test(trimmed)) {
      flowClass = "flow-tool";
    }
    if (isHeader || /^\[(LOCAL|SANDBOX|TRADITIONAL)\]\[TX\]\[TOOLS\]/.test(trimmed)) {
      flowClass = `${flowClass} flow-header`;
    }
    return `<div class="${classifyModalLine(line)} ${flowClass}">${escapeHtml(line) || "&nbsp;"}</div>`;
  });

  els.modalContent.innerHTML = html.join("");
}

function setStatus(text, cls) {
  if (!els.statusPill) return;
  els.statusPill.textContent = text;
  els.statusPill.className = `pill ${cls}`;
}

function setButtonsDisabled(disabled) {
  [els.runLocalBtn, els.runSandboxBtn, els.runTraditionalBtn, els.runAllBtn].forEach((btn) => {
    if (!btn) return;
    btn.disabled = disabled;
  });
}

function fmtTs(ms) {
  const d = new Date(ms);
  return d.toLocaleTimeString([], { hour12: false }) + "." + String(d.getMilliseconds()).padStart(3, "0");
}

function renderLogs(logs) {
  if (!els.logStream) return;
  els.logStream.innerHTML = "";
  if (!logs || !logs.length) {
    els.logStream.innerHTML = `<div class="log-line"><div class="log-ts">--:--</div><div class="log-msg">No logs.</div></div>`;
    return;
  }
  for (const item of logs) {
    const row = document.createElement("div");
    row.className = `log-line ${item.level || "info"}`;
    row.innerHTML = `<div class="log-ts">${fmtTs(item.ts_ms)}</div><div class="log-msg"></div>`;
    row.querySelector(".log-msg").textContent = item.message;
    els.logStream.appendChild(row);
  }
  els.logStream.scrollTop = els.logStream.scrollHeight;
}

function renderLogsInto(container, logs) {
  if (!container) return;
  container.innerHTML = "";
  if (!logs || !logs.length) {
    container.innerHTML = `<div class="log-line"><div class="log-ts">--:--</div><div class="log-msg">No logs.</div></div>`;
    return;
  }
  for (const item of logs) {
    const row = document.createElement("div");
    row.className = `log-line ${item.level || "info"}`;
    row.innerHTML = `<div class="log-ts">${fmtTs(item.ts_ms)}</div><div class="log-msg"></div>`;
    row.querySelector(".log-msg").textContent = item.message;
    container.appendChild(row);
  }
  container.scrollTop = container.scrollHeight;
}

function openModal(title, content, mode = "") {
  setText(els.modalTitle, title || "Expanded View");
  renderModalContent(content || "");
  if (els.modalShell) {
    els.modalShell.classList.remove("mode-local", "mode-sandbox", "mode-traditional");
    const laneCls = laneModeClass(mode);
    if (laneCls) els.modalShell.classList.add(`mode-${laneCls}`);
  }
  if (!els.modalOverlay) return;
  els.modalOverlay.classList.remove("hidden");
  els.modalOverlay.setAttribute("aria-hidden", "false");
}

function closeModal() {
  if (!els.modalOverlay) return;
  els.modalOverlay.classList.add("hidden");
  els.modalOverlay.setAttribute("aria-hidden", "true");
  if (els.modalShell) {
    els.modalShell.classList.remove("mode-local", "mode-sandbox", "mode-traditional");
  }
}

function pretty(obj) {
  if (obj == null) return "null";
  try {
    return JSON.stringify(obj, null, 2);
  } catch {
    return String(obj);
  }
}

function renderResult(data, httpOk) {
  setStatus(data.ok ? "Success" : "Error", data.ok ? "ok" : "error");
  setText(els.metricMode, data.mode || "-");
  setText(els.metricDuration, `${data.duration_ms ?? "-"} ms`);
  setText(els.metricModel, data.metadata?.model || "-");
  const totalTokens =
    data.token_usage?.totals?.total_tokens ??
    (typeof data.token_usage?.totals?.input_tokens === "number" &&
    typeof data.token_usage?.totals?.output_tokens === "number"
      ? data.token_usage.totals.input_tokens + data.token_usage.totals.output_tokens
      : "-");
  setText(els.metricTokens, String(totalTokens));
  setText(els.metricContainer, data.metadata?.container_id || "-");
  setText(els.tokenUsage, pretty(data.token_usage || {}));
  setText(els.metadataBox, pretty(data.metadata || {}));
  setText(els.generatedCode, data.generated_code || "No generated code captured for this run.");
  setText(els.finalText, data.final_text || data.error || "No final text.");
  renderLogs(data.logs || []);

  if (!httpOk && data.error) {
    setText(els.finalText, `${data.error}\n\n${els.finalText?.textContent || ""}`);
  }
}

function laneStatus(mode, text, cls) {
  const lane = laneEls[mode];
  if (!lane || !lane.status) return;
  lane.status.textContent = text;
  lane.status.className = `pill ${cls}`;
}

function laneTokenTotal(data) {
  return String(
    data?.token_usage?.totals?.total_tokens ??
    ((typeof data?.token_usage?.totals?.input_tokens === "number" &&
      typeof data?.token_usage?.totals?.output_tokens === "number")
      ? data.token_usage.totals.input_tokens + data.token_usage.totals.output_tokens
      : "-")
  );
}

function renderLane(mode, data, httpOk) {
  const lane = laneEls[mode];
  if (!lane) return;
  laneStatus(mode, data.ok ? "Success" : "Error", data.ok ? "ok" : "error");
  setText(lane.duration, `${data.duration_ms ?? "-"} ms`);
  setText(lane.tokens, laneTokenTotal(data));
  setText(lane.promptPreview, `Prompt: ${data.prompt_preview || "No prompt preview."}`);
  setText(lane.prompt, data.prompt_text || "No prompt captured.");
  setText(lane.snippet, data.integration_snippet || "No integration snippet captured.");
  setText(lane.code, data.generated_code || "No generated code captured for this run.");
  setText(lane.final, data.final_text || data.error || "No final text.");
  setText(lane.transcript, data.llm_transcript || "No transcript captured for this run.");
  renderLogsInto(lane.logs, data.logs || []);
  if (!httpOk && data.error) {
    setText(lane.final, `${data.error}\n\n${lane.final?.textContent || ""}`);
  }
}

function startLane(mode) {
  const lane = laneEls[mode];
  if (!lane) return;
  laneStatus(mode, "Running...", "running");
  setText(lane.duration, "...");
  setText(lane.tokens, "...");
  setText(lane.promptPreview, "Prompt: Running...");
  setText(lane.prompt, "Running...");
  setText(lane.snippet, "Running...");
  setText(lane.code, "Running...");
  setText(lane.final, "Running...");
  setText(lane.transcript, "Running...");
  renderLogsInto(lane.logs, [
    { ts_ms: Date.now(), level: "info", message: `[UI] Starting ${mode}` },
  ]);
}

let syncingLaneSections = false;

function setupLaneSectionSyncAndMaximize() {
  const detailsList = Array.from(document.querySelectorAll(".lane-section[data-sync]"));
  for (const detailsEl of detailsList) {
    const summary = detailsEl.querySelector("summary");
    if (!summary) continue;

    if (!summary.dataset.enhanced) {
      const originalLabel = summary.textContent?.trim() || "Section";
      summary.textContent = "";

      const titleSpan = document.createElement("span");
      titleSpan.className = "summary-title";
      titleSpan.textContent = originalLabel;

      const actions = document.createElement("span");
      actions.className = "summary-actions";

      const maxBtn = document.createElement("button");
      maxBtn.type = "button";
      maxBtn.className = "maximize-btn";
      maxBtn.textContent = "Maximize";
      maxBtn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        const laneRoot = detailsEl.closest(".swimlane");
        const laneTitle = laneRoot?.querySelector("h3")?.textContent || "Lane";
        const laneMode = laneRoot?.dataset?.lane || "";
        const sectionTitle = titleSpan.textContent || "Section";
        const contentEl = sectionContentEl(detailsEl);
        openModal(`${laneTitle} • ${sectionTitle}`, contentEl?.innerText || "", laneMode);
      });

      actions.appendChild(maxBtn);
      summary.appendChild(titleSpan);
      summary.appendChild(actions);
      summary.dataset.enhanced = "1";
    }

    detailsEl.addEventListener("toggle", () => {
      if (syncingLaneSections) return;
      const syncKey = detailsEl.dataset.sync;
      if (!syncKey) return;
      syncingLaneSections = true;
      try {
        const peers = document.querySelectorAll(`.lane-section[data-sync="${syncKey}"]`);
        peers.forEach((peer) => {
          if (peer === detailsEl) return;
          peer.open = detailsEl.open;
        });
      } finally {
        syncingLaneSections = false;
      }
    });
  }
}

async function runMode(mode) {
  setButtonsDisabled(true);
  setStatus("Running...", "running");
  setText(els.metricMode, mode);
  setText(els.metricDuration, "...");
  setText(els.metricTokens, "...");
  setText(els.generatedCode, "Running...");
  setText(els.finalText, "Running...");
  renderLogs([{ ts_ms: Date.now(), level: "info", message: `[UI] Sending request for mode=${mode}` }]);
  startLane(mode);

  try {
    const body = {};
    const model = (els.modelInput?.value || "").trim();
    if (model) body.model = model;

    const res = await fetch(`/api/run/${mode}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    renderResult(data, res.ok);
    renderLane(mode, data, res.ok);
  } catch (err) {
    setStatus("Error", "error");
    const message = err?.message || String(err);
    renderLogs([{ ts_ms: Date.now(), level: "error", message: `[UI] Request failed: ${message}` }]);
    setText(els.finalText, message);
    setText(els.generatedCode, "No generated code.");
    setText(els.tokenUsage, "{}");
    setText(els.metadataBox, "{}");
    renderLane(mode, {
      ok: false,
      mode,
      duration_ms: 0,
      token_usage: {},
      prompt_preview: "Unavailable due to UI request failure",
      prompt_text: "Unavailable due to UI request failure",
      integration_snippet: "Unavailable due to UI request failure",
      generated_code: "No generated code.",
      final_text: message,
      llm_transcript: "Unavailable due to UI request failure",
      error: message,
      logs: [{ ts_ms: Date.now(), level: "error", message: `[UI] Request failed: ${message}` }],
    }, false);
  } finally {
    setButtonsDisabled(false);
  }
}

async function fetchMode(mode) {
  const body = {};
  const model = (els.modelInput?.value || "").trim();
  if (model) body.model = model;
  const res = await fetch(`/api/run/${mode}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  return { res, data };
}

async function runAllParallel() {
  const modes = ["local_langgraph", "sandbox_langgraph", "traditional"];
  setButtonsDisabled(true);
  setStatus("Running All...", "running");
  renderLogs([{ ts_ms: Date.now(), level: "info", message: "[UI] Running all 3 modes in parallel" }]);
  for (const mode of modes) startLane(mode);

  const tasks = modes.map(async (mode) => {
    try {
      const { res, data } = await fetchMode(mode);
      renderLane(mode, data, res.ok);
      return { mode, ok: res.ok && data.ok, data, resOk: res.ok };
    } catch (err) {
      const message = err?.message || String(err);
      const data = {
        ok: false,
        mode,
        duration_ms: 0,
        token_usage: {},
        prompt_preview: "Unavailable due to UI request failure",
        prompt_text: "Unavailable due to UI request failure",
        integration_snippet: "Unavailable due to UI request failure",
        generated_code: "No generated code.",
        final_text: message,
        llm_transcript: "Unavailable due to UI request failure",
        error: message,
        logs: [{ ts_ms: Date.now(), level: "error", message: `[UI] Request failed: ${message}` }],
      };
      renderLane(mode, data, false);
      return { mode, ok: false, data, resOk: false };
    }
  });

  const results = await Promise.all(tasks);
  const priority = { sandbox_langgraph: 0, local_langgraph: 1, traditional: 2 };
  results.sort((a, b) => priority[a.mode] - priority[b.mode]);
  const firstForMain = results.find((r) => r.mode === "sandbox_langgraph") || results[0];
  if (firstForMain) {
    renderResult(firstForMain.data, firstForMain.resOk);
  }

  const allOk = results.every((r) => r.ok);
  setStatus(allOk ? "All Success" : "Completed (Some Errors)", allOk ? "ok" : "error");
  const summaryLine = results.map((r) => {
    const t = r.data?.token_usage?.totals?.total_tokens ?? "-";
    return `${r.mode}: tokens=${t}, ok=${!!r.ok}`;
  }).join(" | ");
  renderLogs([{ ts_ms: Date.now(), level: "info", message: `[UI] Parallel run complete: ${summaryLine}` }]);
  setButtonsDisabled(false);
}

els.runLocalBtn?.addEventListener("click", () => runMode("local_langgraph"));
els.runSandboxBtn?.addEventListener("click", () => runMode("sandbox_langgraph"));
els.runTraditionalBtn?.addEventListener("click", () => runMode("traditional"));
els.runAllBtn?.addEventListener("click", () => runAllParallel());
els.clearLogsBtn?.addEventListener("click", () => renderLogs([]));
els.modalCloseBtn?.addEventListener("click", closeModal);
els.modalOverlay?.addEventListener("click", (e) => {
  if (e.target === els.modalOverlay) closeModal();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeModal();
});

const missingIds = Object.entries(els)
  .filter(([, el]) => el == null)
  .map(([key]) => key);
if (missingIds.length) {
  console.warn("[UI] Missing DOM elements:", missingIds);
}

setStatus("Idle", "neutral");
renderLogs([]);
Object.keys(laneEls).forEach((mode) => laneStatus(mode, "Idle", "neutral"));
setupLaneSectionSyncAndMaximize();
