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
const surfaceHost = document.getElementById("surface-host");
const surfaceMeta = document.getElementById("surface-meta");
const form = document.getElementById("composer");
const input = document.getElementById("question");
const sendButton = document.getElementById("send");
const textOnlyButton = document.getElementById("text-only");
const protocolBadge = document.getElementById("protocol-badge");
const demoBadge = document.getElementById("demo-badge");

// Une conversation = une surface : on repart d'une conversation neuve à chaque
// chargement de page, sinon l'agent mettrait à jour des surfaces que ce renderer
// ne possède pas (le protocole impose un surfaceId unique par renderer).
let conversationId = null;
let catalogInfo = null;
let textOnly = false;
let busy = false;

const renderer = new A2uiRenderer({
  container: surfaceHost,
  onAction: handleAction,
});

// ── Utilitaires DOM ────────────────────────────────────────────────────────

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function addMessage(role, text) {
  const wrapper = el("article", `bubble is-${role}`);
  const meta = el("div", "bubble-role", role === "user" ? "Vous" : "Agent LPO");
  const body = el("div", "bubble-text");
  body.textContent = text;
  wrapper.append(meta, body);
  chatLog.append(wrapper);
  chatLog.scrollTop = chatLog.scrollHeight;
  return wrapper;
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
      `Traçabilité — ${trace.length} appel(s) MCP · ${total}\u202fms · intention « ${turn.intent?.kind || "?"} »`,
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
    item.append(el("span", "trace-summary", entry.ok ? entry.summary : entry.error || "erreur"));
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

  if (!turn.messages?.length) return;
  if (textOnly) {
    bubble.append(el("p", "text-only-note", "Mode texte seul : surface A2UI non rendue."));
    return;
  }

  for (const message of turn.messages) {
    const accepted = renderer.apply(message);
    if (!accepted) console.warn("Message A2UI refusé", message);
  }
  const kinds = { create: "surface créée", update: "données mises à jour", none: "aucune surface" };
  surfaceMeta.textContent = `${turn.surfaceId || "—"} · ${kinds[turn.mode] || turn.mode} · ${turn.kind}`;
  if (echo) surfaceHost.scrollTop = 0;
}

// ── Appels réseau ──────────────────────────────────────────────────────────

// Jeton optionnel : l'API peut être protégée (LPO_WEB_TOKEN). Il est fourni une
// fois par l'URL (`?token=…`), gardé en mémoire, puis retiré de la barre
// d'adresse pour ne pas rester dans l'historique ni dans les journaux du proxy.
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

async function api(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(body),
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
  busy = true;
  sendButton.disabled = true;
  addMessage("user", question);
  const pending = addMessage("assistant", "Interrogation du serveur de données…");
  pending.classList.add("is-pending");
  try {
    const turn = await api("/api/ask", { question, conversationId });
    pending.remove();
    applyTurn(turn, { echo: true });
  } catch (error) {
    pending.remove();
    addMessage(
      "assistant",
      `Je ne peux pas répondre : ${error.message}. Aucun chiffre n'est affiché sans données vérifiées.`,
    ).classList.add("is-degraded");
  } finally {
    busy = false;
    sendButton.disabled = false;
    input.focus();
  }
}

async function handleAction(action) {
  if (busy) return;
  busy = true;
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
    surfaceMeta.textContent = `${turn.surfaceId} · ${turn.mode === "update" ? "données mises à jour" : "surface recréée"} · ${turn.kind}`;
  } catch (error) {
    addMessage("assistant", `Le filtre n'a pas pu être appliqué : ${error.message}`).classList.add(
      "is-degraded",
    );
  } finally {
    busy = false;
  }
}

// ── Initialisation ─────────────────────────────────────────────────────────

async function loadCatalog() {
  try {
    const info = await fetch("/api/catalog", { headers: authHeaders() }).then((response) =>
      response.json(),
    );
    catalogInfo = info;
    protocolBadge.textContent = `A2UI ${info.protocolVersion} · ${info.components.length} composants`;
    protocolBadge.title = `${info.catalogId}\n${info.protocolDocs}`;
    renderer.catalog = info.catalog;
  } catch (error) {
    protocolBadge.textContent = "catalogue indisponible";
  }
}

async function loadDataset() {
  try {
    const overview = await fetch("/api/dataset", { headers: authHeaders() }).then((response) =>
      response.json(),
    );
    if (overview.is_demo) {
      demoBadge.hidden = false;
      demoBadge.title = overview.dataset_label || "Jeu de démonstration synthétique";
    }
    demoBadge.textContent = overview.is_demo ? "Démo — données synthétiques" : "Données réelles";
    demoBadge.hidden = false;
    renderer.loadGeo();
  } catch (error) {
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
  textOnlyButton.classList.toggle("is-active", textOnly);
  if (textOnly) {
    surfaceHost.replaceChildren(
      el(
        "div",
        "welcome",
        "Mode texte seul actif : les réponses restent lisibles sans renderer A2UI.",
      ),
    );
  }
});

loadCatalog();
loadDataset();
input.focus();
