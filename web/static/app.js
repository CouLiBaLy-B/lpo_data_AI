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
const demoBadge = document.getElementById("demo-badge");
const tokenBadge = document.getElementById("token-badge");

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
    const overview = await fetch("/api/dataset", { headers: authHeaders() }).then((response) =>
      response.json(),
    );
    demoBadge.hidden = false;
    demoBadge.textContent = overview.is_demo ? "Démo — données synthétiques" : "Données réelles";
    demoBadge.title = overview.dataset_label || overview.summary || "";
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
loadDataset();
if (window.matchMedia("(min-width: 641px)").matches) input.focus();
