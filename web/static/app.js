/**
 * Côté client : conversation + application des messages A2UI.
 *
 * Le client est volontairement mince : il n'interprète pas la question, ne
 * calcule aucune donnée, et ne rend que les messages validés envoyés par
 * l'agent. C'est ce qui garantit que les chiffres affichés proviennent du
 * serveur MCP et non du navigateur.
 */

import { A2uiRenderer } from "/static/renderer.js";

const chatLog = document.getElementById("chat-log");
const chatCount = document.getElementById("chat-count");
const chatEmpty = document.getElementById("chat-empty");
const surfaceHost = document.getElementById("surface-host");
const surfaceMeta = document.getElementById("surface-meta");
const panel = document.getElementById("reponse");
const progress = document.getElementById("progress");
const form = document.getElementById("composer");
const input = document.getElementById("question");
const sendButton = document.getElementById("send");
const textOnlyButton = document.getElementById("text-only");
const protocolBadge = document.getElementById("protocol-badge");
const mcpBadge = document.getElementById("mcp-badge");
const llmBadge = document.getElementById("llm-badge");
const demoBadge = document.getElementById("demo-badge");
const tokenBadge = document.getElementById("token-badge");

// Configuration : panneau modal (modèle + serveur MCP).
const settingsOpenButton = document.getElementById("settings-open");
const settingsOverlay = document.getElementById("settings-overlay");
const settingsClose = document.getElementById("settings-close");
const settingsForm = document.getElementById("settings-form");
const settingsStatus = document.getElementById("settings-status");
const settingsSummaryMcp = document.getElementById("settings-summary-mcp");
const settingsSummaryDataset = document.getElementById("settings-summary-dataset");
const settingsSummaryLlm = document.getElementById("settings-summary-llm");
const settingsMode = document.getElementById("settings-llm-mode");
const settingsProvider = document.getElementById("settings-provider");
const settingsModel = document.getElementById("settings-model");
const settingsApiKey = document.getElementById("settings-api-key");
const settingsKeyHint = document.getElementById("settings-key-hint");
const settingsTransport = document.getElementById("settings-mcp-transport");
const settingsHttpFields = document.getElementById("settings-http-fields");
const settingsStdioFields = document.getElementById("settings-stdio-fields");
const settingsUrl = document.getElementById("settings-mcp-url");
const settingsCommand = document.getElementById("settings-mcp-command");
const settingsArgs = document.getElementById("settings-mcp-args");
const settingsSave = document.getElementById("settings-save");
const settingsTestButton = document.getElementById("settings-test");
const settingsResetButton = document.getElementById("settings-reset");
const welcomeMarkup = document.getElementById("surface-welcome")?.outerHTML || "";

// Une conversation = une surface : on repart d'une conversation neuve à chaque
// chargement de page, sinon l'agent mettrait à jour des surfaces que ce renderer
// ne possède pas (le protocole impose un surfaceId unique par renderer).
let conversationId = null;
let textOnly = false;
let busy = false;
let turns = 0;

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

const renderer = new A2uiRenderer({
  container: surfaceHost,
  onAction: handleAction,
});

// ── Jeton d'accès optionnel ────────────────────────────────────────────────
// Fourni une fois par l'URL (`?token=…`), puis retiré de la barre d'adresse pour
// ne pas rester dans l'historique ni dans les journaux du proxy.

const accessToken = (() => {
  const url = new URL(window.location.href);
  const token = url.searchParams.get("token");
  if (token) {
    url.searchParams.delete("token");
    window.history.replaceState(null, "", url.toString());
  }
  return token || "";
})();

function authHeaders(extra = {}) {
  return accessToken ? { ...extra, Authorization: `Bearer ${accessToken}` } : extra;
}

// ── Utilitaires DOM ────────────────────────────────────────────────────────

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function scrollToEnd(node) {
  const top = node.scrollHeight;
  // Les moteurs de test (jsdom) n'implémentent pas `scrollTo` : le repli
  // silencieux évite qu'une absence d'animation casse la conversation.
  if (typeof node.scrollTo === "function") {
    node.scrollTo({ top, behavior: reduceMotion.matches ? "auto" : "smooth" });
  } else {
    node.scrollTop = top;
  }
}

function revealPanel() {
  if (typeof panel.scrollIntoView !== "function") return;
  panel.scrollIntoView({ behavior: reduceMotion.matches ? "auto" : "smooth", block: "nearest" });
}

function setBusy(state) {
  busy = state;
  sendButton.disabled = state;
  panel.setAttribute("aria-busy", String(state));
  progress.hidden = !state;
  progress.dataset.active = String(state);
  sendButton.textContent = state ? "Envoi…" : "Envoyer";
}

function updateCount() {
  chatCount.textContent = turns === 0 ? "0 tour" : `${turns} tour${turns > 1 ? "s" : ""}`;
}

function addMessage(role, text) {
  if (chatEmpty && chatEmpty.isConnected) chatEmpty.remove();
  const wrapper = el("article", `bubble is-${role}`);
  const meta = el("p", "bubble-role", role === "user" ? "Vous" : "Agent LPO");
  const body = el("div", "bubble-text");
  body.textContent = text;
  wrapper.append(meta, body);
  chatLog.append(wrapper);
  scrollToEnd(chatLog);
  if (role === "assistant") {
    turns += 1;
    updateCount();
  }
  return wrapper;
}

function setBubbleText(bubble, text) {
  const body = bubble.querySelector(".bubble-text");
  if (body) body.textContent = text;
  scrollToEnd(chatLog);
}

function renderTrace(turn, anchor) {
  const trace = turn.trace || [];
  if (!trace.length) return;
  const details = el("details", "trace");
  const total = trace.reduce((sum, entry) => sum + (entry.duration_ms || 0), 0);
  details.append(
    el(
      "summary",
      "",
      `${trace.length} appel${trace.length > 1 ? "s" : ""} MCP · ${total}\u202fms · intention « ${
        turn.intent?.kind || "?"
      } »`,
    ),
  );
  const list = el("ol", "trace-list");
  for (const entry of trace) {
    const item = el("li", entry.ok ? "trace-ok" : "trace-error");
    const args = Object.entries(entry.arguments || {})
      .map(([key, value]) => `${key}=${Array.isArray(value) ? value.join("|") : value}`)
      .join(", ");
    item.append(el("code", "", entry.tool));
    item.append(el("span", "trace-args", args));
    item.append(
      el(
        "span",
        "trace-status",
        entry.ok ? `${entry.duration_ms}\u202fms` : `échec · ${entry.error || "inconnu"}`,
      ),
    );
    if (entry.ok && entry.summary) item.append(el("span", "trace-summary", entry.summary));
    list.append(item);
  }
  if (turn.validation) {
    const status = el(
      "li",
      turn.validation.ok ? "trace-ok" : "trace-error",
      turn.validation.ok
        ? `A2UI validé : ${turn.validation.checked} message(s)`
        : `A2UI rejeté : ${(turn.validation.errors || []).join(" ; ")}`,
    );
    status.append(el("span", "trace-status", turn.validation.ok ? "ok" : "rejeté"));
    list.append(status);
  }
  details.append(list);
  anchor.append(details);
}

// ── Rendu d'un tour ────────────────────────────────────────────────────────

function applyTurn(turn, { echo = false } = {}) {
  if (turn.conversationId) conversationId = turn.conversationId;

  const bubble = addMessage("assistant", turn.text || "(pas de résumé)");
  if (turn.degraded) bubble.classList.add("is-degraded");
  renderTrace(turn, bubble);

  // Le bandeau de réponse décrit ce que la surface contient, même quand il n'y
  // a rien à rendre : un état vide explicite vaut mieux qu'un bandeau périmé.
  const kinds = { create: "surface créée", update: "données mises à jour", none: "aucune surface" };
  surfaceMeta.textContent = [turn.surfaceId || "—", kinds[turn.mode] || turn.mode, turn.kind]
    .filter(Boolean)
    .join(" · ");

  if (!turn.messages?.length) return;
  if (textOnly) {
    bubble.append(
      el("p", "text-only-note", "Mode texte seul : la surface A2UI n'est pas rendue."),
    );
    return;
  }

  for (const message of turn.messages) {
    const accepted = renderer.apply(message);
    if (!accepted) console.warn("Message A2UI refusé", message);
  }
  // La première surface remplace l'écran d'accueil : il n'a plus de raison
  // d'occuper la page une fois la réponse affichée.
  document.getElementById("surface-welcome")?.remove();
  if (echo) revealPanel();
}

// ── Appels réseau ──────────────────────────────────────────────────────────

async function api(path, body, { method = "POST" } = {}) {
  const response = await fetch(path, {
    method,
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch (_) {
      /* réponse non JSON */
    }
    throw new Error(detail);
  }
  return response.json();
}

async function ask(question) {
  if (busy) return;
  setBusy(true);
  addMessage("user", question);
  const pending = addMessage("assistant", "Interrogation du serveur de données…");
  pending.classList.add("is-pending");
  try {
    const turn = await api("/api/ask", { question, conversationId });
    pending.remove();
    turns = Math.max(0, turns - 1);
    updateCount();
    applyTurn(turn, { echo: true });
  } catch (error) {
    pending.remove();
    turns = Math.max(0, turns - 1);
    updateCount();
    addMessage(
      "assistant",
      `Réponse indisponible : ${error.message}. Aucun chiffre n'est affiché sans données vérifiées.`,
    ).classList.add("is-error");
  } finally {
    setBusy(false);
    input.focus();
  }
}

async function handleAction(action) {
  if (busy) return;
  setBusy(true);
  try {
    const turn = await api("/api/action", {
      conversationId,
      surfaceId: action.surfaceId,
      action: { name: action.name, context: action.context },
      dataModel: action.dataModel,
    });
    const bubble = addMessage(
      "assistant",
      (action.userMessage ? `${action.userMessage} — ` : "") + turn.text,
    );
    renderTrace(turn, bubble);
    for (const message of turn.messages || []) renderer.apply(message);
    surfaceMeta.textContent = [
      turn.surfaceId,
      turn.mode === "update" ? "données mises à jour" : "surface recréée",
      turn.kind,
    ]
      .filter(Boolean)
      .join(" · ");
  } catch (error) {
    addMessage("assistant", `Le filtre n'a pas pu être appliqué : ${error.message}`).classList.add(
      "is-error",
    );
  } finally {
    setBusy(false);
  }
}

// ── Configuration (modèle + serveur MCP) ──────────────────────────────────

let settingsSnapshot = null;

function setSettingsStatus(text, tone = "info") {
  settingsStatus.textContent = text;
  settingsStatus.dataset.tone = tone;
}

function openSettings() {
  settingsOverlay.hidden = false;
  settingsOpenButton.setAttribute("aria-expanded", "true");
  loadSettings();
  settingsClose.focus();
}

function closeSettings() {
  settingsOverlay.hidden = true;
  settingsOpenButton.setAttribute("aria-expanded", "false");
  settingsOpenButton.focus();
}

function providerLabel(provider) {
  return provider.ready ? provider.label : `${provider.label} — clé absente`;
}

function fillProviders(providers, selected) {
  settingsProvider.replaceChildren();
  for (const provider of providers || []) {
    const option = el("option", "", providerLabel(provider));
    option.value = provider.id;
    if (provider.id === selected) option.selected = true;
    settingsProvider.append(option);
  }
}

function syncMcpFields() {
  const http = settingsTransport.value === "http";
  settingsHttpFields.hidden = !http;
  settingsStdioFields.hidden = http;
  settingsUrl.disabled = !http;
  settingsCommand.disabled = http;
  settingsArgs.disabled = http;
}

function datasetSummary(dataset) {
  if (!dataset || !dataset.profile) return "jeu de données inconnu";
  const records = dataset.recordsLabel || "enregistrements";
  const name = dataset.label || dataset.profile;
  const rows = typeof dataset.totalRows === "number" ? ` · ${dataset.totalRows} ${records}` : "";
  const kind = dataset.isDemo ? "démonstration" : "réelles";
  return `${name} (${dataset.profile}) · données ${kind}${rows}`;
}

function applySettings(settings) {
  settingsSnapshot = settings;
  settingsMode.value = settings.llm.mode;
  fillProviders(settings.llm.providers, settings.llm.provider);
  settingsModel.value = settings.llm.model || "";
  settingsApiKey.value = "";
  settingsKeyHint.textContent = settings.llm.keyEnv
    ? `${settings.llm.keyEnv} : ${
        settings.llm.apiKeySet ? "clé disponible" : "aucune clé"
      } — la clé saisie reste sur le serveur et n'est jamais renvoyée.`
    : "Fournisseur local : aucune clé requise.";

  const mcp = settings.mcp;
  settingsTransport.value = mcp.transport;
  settingsUrl.value = mcp.url || "";
  settingsCommand.value = mcp.command || "";
  settingsArgs.value = (mcp.args || []).join("\n");
  syncMcpFields();

  const status = mcp.status;
  if (status && status.ok === false) {
    settingsSummaryMcp.textContent = `${mcp.target} — injoignable : ${status.error || "erreur"}`;
  } else {
    const tools = status && Array.isArray(status.tools) ? status.tools.length : null;
    settingsSummaryMcp.textContent = `${mcp.target} (${mcp.transport})${
      tools === null ? "" : ` · ${tools} outil${tools > 1 ? "s" : ""}`
    }`;
  }
  const llmLabel =
    settings.llm.mode === "off"
      ? "heuristique déterministe (mode off)"
      : `${settings.llm.provider}/${settings.llm.model || "modèle non renseigné"} (${
          settings.llm.apiKeySet ? "clé disponible" : "clé absente"
        })`;
  settingsSummaryLlm.textContent = llmLabel;
  if (!settings.editable) {
    settingsSave.disabled = true;
    settingsTestButton.disabled = true;
    settingsResetButton.disabled = true;
    setSettingsStatus(
      "Configuration figée par l'environnement (LPO_WEB_SETTINGS=off) : lecture seule.",
      "warning",
    );
  }
}

async function loadSettings() {
  try {
    const settings = await api("/api/settings", undefined, { method: "GET" });
    applySettings(settings);
  } catch (error) {
    setSettingsStatus(`Réglages indisponibles : ${error.message}`, "error");
  }
  try {
    const overview = await api("/api/dataset", undefined, { method: "GET" });
    settingsSummaryDataset.textContent = datasetSummary({
      label: overview.dataset_label,
      profile: overview.profile,
      totalRows: overview.total_rows,
      recordsLabel: overview.records_label,
      isDemo: overview.is_demo,
    });
  } catch (error) {
    settingsSummaryDataset.textContent = `indisponible : ${error.message}`;
  }
}

function patchFromForm() {
  const patch = {
    llm: {
      mode: settingsMode.value,
      provider: settingsProvider.value,
      model: settingsModel.value.trim(),
    },
    mcp: { transport: settingsTransport.value },
  };
  const key = settingsApiKey.value.trim();
  if (key) patch.llm.apiKey = key;
  if (settingsTransport.value === "http") {
    patch.mcp.url = settingsUrl.value.trim();
  } else {
    patch.mcp.command = settingsCommand.value.trim();
    patch.mcp.args = settingsArgs.value
      .split("\n")
      .map((argument) => argument.trim())
      .filter(Boolean);
  }
  return patch;
}

function resetConversation() {
  conversationId = null;
  turns = 0;
  updateCount();
  renderer.reset();
  chatLog.replaceChildren();
  if (welcomeMarkup) surfaceHost.innerHTML = welcomeMarkup;
  surfaceMeta.textContent = "aucune surface";
}

async function refreshStatus() {
  try {
    const health = await api("/api/health", undefined, { method: "GET" });
    const mcp = health.mcp || {};
    mcpBadge.textContent = mcp.ok
      ? `MCP · ${(mcp.tools || []).length} outils`
      : "MCP injoignable";
    mcpBadge.classList.toggle("badge-error", !mcp.ok);
    mcpBadge.title = [mcp.target, mcp.error].filter(Boolean).join("\n");

    const llm = health.llm || {};
    llmBadge.hidden = false;
    if (!llm.enabled || llm.mode === "off") {
      llmBadge.textContent = "Intention · heuristique";
    } else if (!llm.ready) {
      llmBadge.textContent = "Intention · LLM sans clé";
    } else {
      llmBadge.textContent = `Intention · ${llm.provider}`;
    }
    llmBadge.title = [
      `Mode : ${llm.mode}`,
      `Fournisseur : ${llm.provider}`,
      `Modèle : ${llm.model || "—"}`,
      `Réglages : ${health.settingsSource === "ui" ? "interface" : "environnement"}`,
    ].join("\n");
  } catch (error) {
    console.warn("état du service indisponible", error);
  }
}

async function saveSettings(event) {
  event.preventDefault();
  settingsSave.disabled = true;
  setSettingsStatus("Application des réglages…", "info");
  try {
    const result = await api("/api/settings", patchFromForm());
    applySettings({ ...result.settings, mcp: { ...result.settings.mcp, status: result.mcp } });
    const parts = [];
    if (result.restarted.includes("mcp")) {
      const tools = (result.mcp && result.mcp.tools ? result.mcp.tools.length : 0);
      parts.push(`serveur MCP connecté (${tools} outil${tools > 1 ? "s" : ""})`);
      resetConversation();
    }
    if (result.restarted.includes("llm")) parts.push("modèle mis à jour");
    setSettingsStatus(
      parts.length
        ? `Enregistré — ${parts.join(", ")}${
            result.restarted.includes("mcp") ? ". Nouvelle conversation" : ""
          }.`
        : "Enregistré — aucun changement effectif.",
      "ok",
    );
    await refreshStatus();
    await loadDataset();
  } catch (error) {
    setSettingsStatus(`Refusé : ${error.message}`, "error");
  } finally {
    settingsSave.disabled = !(settingsSnapshot ? settingsSnapshot.editable : true);
  }
}

async function testSettings() {
  settingsTestButton.disabled = true;
  setSettingsStatus("Test en cours…", "info");
  try {
    const result = await api("/api/settings/test", {
      target: "mcp",
      mcp: patchFromForm().mcp,
    });
    if (!result.ok) {
      setSettingsStatus(`Échec : ${result.error}`, "error");
      return;
    }
    const tools = (result.tools || []).length;
    setSettingsStatus(
      `Connexion réussie (${result.durationMs} ms) — ${tools} outil${
        tools > 1 ? "s" : ""
      } · ${datasetSummary(result.dataset)}`,
      "ok",
    );
  } catch (error) {
    setSettingsStatus(`Échec : ${error.message}`, "error");
  } finally {
    settingsTestButton.disabled = false;
  }
}

async function resetSettings() {
  settingsResetButton.disabled = true;
  setSettingsStatus("Retour aux valeurs d'environnement…", "info");
  try {
    const result = await api("/api/settings", { reset: "env" });
    applySettings({ ...result.settings, mcp: { ...result.settings.mcp, status: result.mcp } });
    resetConversation();
    setSettingsStatus("Valeurs .env réappliquées — conversation réinitialisée.", "ok");
    await refreshStatus();
    await loadDataset();
  } catch (error) {
    setSettingsStatus(`Retour impossible : ${error.message}`, "error");
  } finally {
    settingsResetButton.disabled = false;
  }
}

settingsOpenButton.addEventListener("click", openSettings);
settingsClose.addEventListener("click", closeSettings);
settingsOverlay.addEventListener("click", (event) => {
  if (event.target === settingsOverlay) closeSettings();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !settingsOverlay.hidden) closeSettings();
});
settingsTransport.addEventListener("change", syncMcpFields);
settingsForm.addEventListener("submit", saveSettings);
settingsTestButton.addEventListener("click", testSettings);
settingsResetButton.addEventListener("click", resetSettings);

// ── Initialisation ─────────────────────────────────────────────────────────

async function loadCatalog() {
  try {
    const info = await fetch("/api/catalog", { headers: authHeaders() }).then((response) =>
      response.json(),
    );
    protocolBadge.textContent = `A2UI ${info.protocolVersion} · ${info.components.length} composants`;
    protocolBadge.title = `${info.catalogId}\n${info.protocolDocs}`;
    renderer.catalog = info.catalog;
  } catch (error) {
    protocolBadge.textContent = "catalogue indisponible";
    protocolBadge.title = String(error);
  }
}

async function loadDataset() {
  try {
    const overview = await api("/api/dataset", undefined, { method: "GET" });
    demoBadge.hidden = false;
    demoBadge.textContent = overview.is_demo ? "Démo — données synthétiques" : "Données réelles";
    demoBadge.title = overview.dataset_label || overview.summary || "";
    renderer.loadGeo();
  } catch (error) {
    demoBadge.hidden = true;
    mcpBadge.textContent = "MCP injoignable";
    mcpBadge.classList.add("badge-error");
    mcpBadge.title = String(error.message || error);
    console.warn("aperçu du jeu de données indisponible", error);
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  ask(question);
});

document.getElementById("examples").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-question]");
  if (!button) return;
  ask(button.dataset.question);
});

textOnlyButton.addEventListener("click", () => {
  textOnly = !textOnly;
  textOnlyButton.setAttribute("aria-pressed", String(textOnly));
  surfaceHost.hidden = textOnly;
  let note = document.querySelector(".text-only-note.surface-level");
  if (textOnly && !note) {
    note = el(
      "p",
      "text-only-note surface-level",
      "Mode texte seul : les réponses restent lisibles sans renderer A2UI. Désactivez le mode pour revoir les surfaces.",
    );
    surfaceHost.insertAdjacentElement("afterend", note);
  } else if (!textOnly && note) {
    note.remove();
  }
});

if (accessToken) tokenBadge.hidden = false;

updateCount();
loadCatalog();
refreshStatus();
loadDataset();
settingsOverlay.hidden = true;
if (window.matchMedia("(min-width: 641px)").matches) input.focus();
