// Rend les surfaces de `turns.json` avec le renderer de production et écrit une
// page HTML autonome : c'est la preuve « messages A2UI → DOM » de l'exemple.
//
// Prérequis : node ≥ 18 et jsdom.
//   mkdir -p .snapshot && cd .snapshot && npm i jsdom && cd ..
//   NODE_PATH=$PWD/.snapshot/node_modules node lpo_a2ui/examples/render_snapshot.mjs
//   → lpo_a2ui/examples/rendered.html  (+ rendered.report.json)
//
// Le script n'appelle aucun serveur : il rejoue les messages sauvegardés par
// `python -m lpo_a2ui.examples.build_transcript` sur le renderer
// `web/static/renderer.js`, exactement comme le ferait le navigateur. Un seul
// renderer est utilisé pour toute la conversation, ce qui reproduit fidèlement
// le comportement « canvas » : les filtres ne font que mettre à jour le data
// model de la surface existante.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { JSDOM } from "jsdom";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "../..");
const STATIC = path.join(ROOT, "web/static");

const data = JSON.parse(fs.readFileSync(path.join(HERE, "turns.json"), "utf8"));
const styles = fs.readFileSync(path.join(STATIC, "styles.css"), "utf8");
const geo = JSON.parse(fs.readFileSync(path.join(STATIC, "geo/departements.geojson"), "utf8"));

const dom = new JSDOM(
  `<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Exemple A2UI — rendu</title>
   <style>${styles}
   /* Habillage de la page d'exemple : mêmes jetons que l'application, pour que
      l'instantané montre le rendu réel et non une maquette à part. */
   body{background:var(--paper);color:var(--ink);font-family:var(--font-ui);margin:0}
   .snapshot{max-width:1100px;margin:0 auto;padding:var(--space-8) var(--space-4) var(--space-10)}
   .snapshot > h1{font-size:20px;font-weight:650;letter-spacing:-.01em}
   .snapshot > .meta{max-width:78ch;margin-top:var(--space-2);color:var(--ink-3);font-size:12.5px}
   .snapshot section{margin-top:var(--space-8);padding-top:var(--space-4);border-top:1px solid var(--line)}
   .snapshot h2{font-size:13px;font-weight:600;font-family:var(--font-mono);letter-spacing:0}
   .snapshot .q{color:var(--ink-2);margin:var(--space-1) 0 var(--space-3);font-size:13.5px}
   .snapshot .frame{background:var(--paper);border:1px solid var(--line);border-radius:var(--radius);padding:var(--space-4)}
   .snapshot .frame > .a2ui-surface-host{display:block}
   .snapshot .live{display:none}
   .snapshot .meta{font-size:12px;color:var(--ink-3)}
   .snapshot .fallback{margin-top:var(--space-3);white-space:pre-wrap;border-left:2px solid var(--line-strong);
     padding:var(--space-2) var(--space-3);color:var(--ink-2);font-size:12.5px;line-height:1.55}
   </style></head><body><div class="snapshot">
   <h1>Exemple bout en bout — surfaces rendues</h1>
   <p class="meta">Messages produits par l'agent, rendus par <code>web/static/renderer.js</code>
   dans jsdom. Aucun HTML n'est fourni par le serveur.</p>
   <div class="live" id="live"></div></div></body></html>`,
  { url: "http://127.0.0.1:8080/", pretendToBeVisual: true }
);

globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.Element = dom.window.Element;
globalThis.HTMLElement = dom.window.HTMLElement;
globalThis.Node = dom.window.Node;
globalThis.NodeFilter = dom.window.NodeFilter;
globalThis.SVGElement = dom.window.SVGElement;

// Le renderer charge le fond de carte par fetch : on le sert depuis le dépôt.
const nativeFetch = globalThis.fetch;
globalThis.fetch = async (url, init) =>
  String(url).includes("geojson")
    ? { ok: true, json: async () => geo }
    : nativeFetch(url, init);

const { A2uiRenderer } = await import(path.join(STATIC, "renderer.js"));

const container = document.querySelector(".snapshot");
const live = document.getElementById("live");
const renderer = new A2uiRenderer({ container: live, onAction: () => {} });

const report = [];
let surfacesRendered = 0;

for (const turn of data.turns) {
  const section = document.createElement("section");
  const title = document.createElement("h2");
  title.textContent = `Tour ${turn.index} — ${turn.kind} (${turn.mode})`;
  const question = document.createElement("p");
  question.className = "q";
  question.textContent =
    turn.action !== null
      ? `Action A2UI : ${JSON.stringify(turn.action)}`
      : `« ${turn.question} »`;
  const frame = document.createElement("div");
  frame.className = "frame";
  section.append(title, question, frame);
  container.append(section);

  const errors = [];
  const ignored = [];
  for (const message of turn.messages ?? []) {
    const key = Object.keys(message).find((k) => k !== "version");
    const accepted = renderer.apply(message);
    if (!accepted) {
      // Supprimer une surface qu'un renderer vierge ne connaît pas est un no-op
      // légitime (la conversation a été rejouée depuis des messages partiels).
      if (key === "deleteSurface") ignored.push(message.deleteSurface.surfaceId);
      else errors.push(`${key} refusé`);
    }
  }
  if (renderer.errors?.length) errors.push(...renderer.errors.map(String));

  // La carte charge son fond GeoJSON de façon asynchrone : on laisse la promesse
  // se résoudre avant de figer le DOM.
  if (renderer.geoPromise) {
    try {
      await renderer.geoPromise;
    } catch (error) {
      errors.push(`fond de carte : ${error}`);
    }
  }

  // Instantané du DOM courant (l'état du renderer persiste d'un tour à l'autre).
  const snapshot = live.cloneNode(true);
  snapshot.removeAttribute("id");
  snapshot.className = "a2ui-surface-host";
  frame.append(snapshot);

  const rendered = snapshot.querySelectorAll(".a2ui-surface");
  surfacesRendered = Math.max(surfacesRendered, rendered.length);
  if (!rendered.length) errors.push("aucune surface dans le DOM");

  for (const node of rendered) {
    if (!node.querySelector("*")) errors.push(`surface ${node.dataset.surface} vide`);
  }

  const surface = renderer.surfaces.get(turn.surfaceId);
  report.push({
    tour: turn.index,
    kind: turn.kind,
    mode: turn.mode,
    surfaceId: turn.surfaceId,
    surfaces: [...renderer.surfaces.keys()],
    composants: surface ? surface.components.size : 0,
    noeuds: snapshot.querySelectorAll("*").length,
    noop: ignored,
    erreurs: errors,
  });

  if (turn.text) {
    const fallback = document.createElement("div");
    fallback.className = "fallback";
    fallback.textContent = turn.text;
    frame.append(fallback);
  }
  if (errors.length) {
    const alert = document.createElement("p");
    alert.className = "meta";
    alert.textContent = `Erreurs de rendu : ${errors.join(" ; ")}`;
    frame.append(alert);
  }
}

const output = path.join(HERE, "rendered.html");
fs.writeFileSync(output, dom.serialize(), "utf8");
fs.writeFileSync(
  path.join(HERE, "rendered.report.json"),
  JSON.stringify({ surfaces_rendered: surfacesRendered, report }, null, 2),
  "utf8"
);

console.log(`Surfaces simultanées max : ${surfacesRendered}`);
for (const entry of report) {
  const status = entry.erreurs.length ? "FAIL" : "OK  ";
  console.log(
    `  [${status}] tour ${entry.tour} — ${entry.kind}/${entry.mode} — ` +
      `${entry.composants} composants, ${entry.noeuds} nœuds DOM` +
      (entry.noop.length ? ` (deleteSurface sans effet : ${entry.noop.join(",")})` : "")
  );
  for (const error of entry.erreurs) console.log(`         ${error}`);
}
if (report.some((entry) => entry.erreurs.length)) process.exitCode = 1;
console.log(`Page écrite : ${output}`);
