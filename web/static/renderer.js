/**
 * Renderer A2UI v1.0 — implémentation de référence minimale, sans dépendance.
 *
 * Périmètre strictement conforme au sous-ensemble utilisé par l'agent :
 *   messages   : createSurface, updateComponents, updateDataModel, deleteSurface
 *   liaisons   : {"@path": "/json/pointer"} (absolu) et {"@path": "relative"}
 *                à l'intérieur d'un template de liste
 *   enfants    : ChildList statique (tableau d'ids) ou template
 *                {path, componentId}
 *   actions    : Action.event -> remontée à l'agent (filtres, boutons)
 *
 * Aucun HTML/CSS/JS n'est reçu du serveur : seuls des messages JSON déclaratifs
 * sont interprétés, et chaque composant est construit avec des API DOM sûres
 * (textContent, createElementNS). Le catalogue (`catalog.json`) est la seule
 * source de vérité sur les composants autorisés.
 *
 * Références : https://a2ui.org/specification/v1.0-a2ui/ (messages, ChildList,
 * DataBinding), https://a2ui.org/concepts/data-binding/.
 */

const SVG_NS = "http://www.w3.org/2000/svg";

/** Accès au data model d'une surface (JSON Pointer, RFC 6901). */
class DataModel {
  constructor(root = {}) {
    this.root = root && typeof root === "object" ? root : {};
  }

  static split(path) {
    return String(path || "")
      .split("/")
      .filter((part) => part.length)
      .map((part) => part.replace(/~1/g, "/").replace(/~0/g, "~"));
  }

  static walk(node, parts) {
    let current = node;
    for (const part of parts) {
      if (current === null || current === undefined) return undefined;
      if (Array.isArray(current)) {
        const index = Number(part);
        if (!Number.isInteger(index)) return undefined;
        current = current[index];
      } else if (typeof current === "object") {
        if (!(part in current)) return undefined;
        current = current[part];
      } else {
        return undefined;
      }
    }
    return current;
  }

  /** Résout un chemin absolu ou relatif à `basePath`. */
  get(path, basePath = "") {
    if (path === undefined || path === null) return undefined;
    const text = String(path);
    if (text.startsWith("/")) return DataModel.walk(this.root, DataModel.split(text));
    const combined = `${basePath || ""}/${text}`;
    return DataModel.walk(this.root, DataModel.split(combined));
  }

  /** Écrit une valeur ; crée les niveaux intermédiaires manquants. */
  set(path, value) {
    const parts = DataModel.split(path);
    if (!parts.length) {
      this.root = value;
      return;
    }
    let current = this.root;
    for (let i = 0; i < parts.length - 1; i += 1) {
      const key = parts[i];
      if (typeof current[key] !== "object" || current[key] === null) {
        current[key] = Number.isInteger(Number(parts[i + 1])) ? [] : {};
      }
      current = current[key];
    }
    current[parts[parts.length - 1]] = value;
  }

  /** Remplace tout le data model (path `/` ou omis). */
  replace(value) {
    this.root = value && typeof value === "object" ? value : {};
  }
}

/** État d'une surface : composants (liste d'adjacence) + data model. */
class Surface {
  constructor(surfaceId, { catalogId, sendDataModel = false, components = [], dataModel = {} }) {
    this.surfaceId = surfaceId;
    this.catalogId = catalogId || "";
    this.sendDataModel = Boolean(sendDataModel);
    this.components = new Map();
    this.data = new DataModel(dataModel || {});
    this.upsert(components);
  }

  upsert(components) {
    for (const component of components || []) {
      if (component && component.id) this.components.set(component.id, component);
    }
  }

  get(id) {
    return this.components.get(id);
  }
}

export class A2uiRenderer {
  /**
   * @param {object} options
   * @param {HTMLElement} options.container  zone d'accueil des surfaces
   * @param {object} options.catalog         catalogue A2UI (composants autorisés)
   * @param {(action: object) => void} options.onAction  remontée des actions
   */
  constructor({ container, catalog = null, onAction = null }) {
    this.container = container;
    this.catalog = catalog || { components: {} };
    this.onAction = onAction;
    this.surfaces = new Map();
    this.geo = null;
    this.geoPromise = null;
    this.errors = [];
  }

  /** Applique un message agent → renderer. Renvoie true si accepté. */
  apply(message) {
    if (!message || typeof message !== "object") return false;
    if (message.createSurface) return this.#createSurface(message.createSurface);
    if (message.updateComponents) return this.#updateComponents(message.updateComponents);
    if (message.updateDataModel) return this.#updateDataModel(message.updateDataModel);
    if (message.deleteSurface) return this.#deleteSurface(message.deleteSurface);
    this.errors.push(`Message A2UI non géré : ${Object.keys(message).join(", ")}`);
    return false;
  }

  #createSurface(payload) {
    const { surfaceId } = payload;
    if (!surfaceId) return false;
    if (this.surfaces.has(surfaceId)) {
      // Le protocole interdit de recréer une surface existante : on la remplace
      // côté client pour rester tolérant aux reconnexions.
      this.surfaces.delete(surfaceId);
    }
    this.surfaces.set(surfaceId, new Surface(surfaceId, payload));
    this.render(surfaceId);
    return true;
  }

  #updateComponents(payload) {
    const surface = this.surfaces.get(payload.surfaceId);
    if (!surface) return false;
    surface.upsert(payload.components);
    this.render(payload.surfaceId);
    return true;
  }

  #updateDataModel(payload) {
    const surface = this.surfaces.get(payload.surfaceId);
    if (!surface) return false;
    const path = payload.path || "/";
    if (path === "/" || path === "") surface.data.replace(payload.value);
    else surface.data.set(path, payload.value);
    this.render(payload.surfaceId);
    return true;
  }

  #deleteSurface(payload) {
    const surface = this.surfaces.get(payload.surfaceId);
    if (!surface) return false;
    this.surfaces.delete(payload.surfaceId);
    const node = this.container.querySelector(`[data-surface="${payload.surfaceId}"]`);
    if (node) node.remove();
    return true;
  }

  reset() {
    this.surfaces.clear();
    this.container.replaceChildren();
  }

  /** Re-rend une surface (reconstruction complète, surfaces de taille modeste). */
  render(surfaceId) {
    const surface = this.surfaces.get(surfaceId);
    if (!surface) return;
    let host = this.container.querySelector(`[data-surface="${surfaceId}"]`);
    if (!host) {
      host = document.createElement("div");
      host.className = "a2ui-surface";
      host.dataset.surface = surfaceId;
      this.container.append(host);
    }
    const root = surface.get("root");
    host.replaceChildren();
    if (!root) {
      host.append(this.#placeholder("Surface sans composant racine."));
      return;
    }
    try {
      host.append(this.#renderComponent(root.id, surface, ""));
    } catch (error) {
      console.error("A2UI render error", error, root);
      host.append(this.#placeholder(`Rendu impossible : ${error.message}`));
    }
  }

  // ── Résolution des liaisons et des enfants ───────────────────────────────

  #isBinding(value) {
    return value && typeof value === "object" && !Array.isArray(value) && "@path" in value;
  }

  #isFunctionCall(value) {
    return value && typeof value === "object" && !Array.isArray(value) && "@call" in value;
  }

  /** Résout une valeur (littérale, liaison, ou fonction non supportée). */
  resolve(value, surface, basePath) {
    if (this.#isBinding(value)) {
      const resolved = surface.data.get(value["@path"], basePath);
      return resolved === undefined ? null : resolved;
    }
    if (this.#isFunctionCall(value)) {
      // Ce catalogue n'expose aucune fonction d'expression : tout est calculé
      // côté serveur, donc une @call est un message invalide pour ce renderer.
      this.errors.push(`Fonction non supportée par le catalogue LPO : ${value["@call"]}`);
      return null;
    }
    return value;
  }

  /** Enfants d'un composant : ids statiques ou instances de template. */
  #children(component, surface, basePath) {
    const list = component.children;
    if (!list) return [];
    if (Array.isArray(list)) {
      return list.map((id) => ({ id, basePath }));
    }
    if (list.path && list.componentId) {
      const items = surface.data.get(list.path, basePath);
      if (!Array.isArray(items)) return [];
      const base = list.path.startsWith("/") ? list.path : `${basePath}/${list.path}`;
      return items.map((_, index) => ({
        id: list.componentId,
        basePath: `${base}/${index}`,
      }));
    }
    return [];
  }

  #childId(component, surface, basePath, prop) {
    const value = this.resolve(component[prop], surface, basePath);
    return typeof value === "string" ? value : null;
  }

  // ── Rendu d'un composant ─────────────────────────────────────────────────

  #renderComponent(id, surface, basePath) {
    const component = surface.get(id);
    if (!component) return this.#placeholder(`Composant « ${id} » manquant (rendu progressif).`);
    const type = component.component;
    const render = this.#registry[type];
    if (!render) return this.#placeholder(`Composant « ${type} » absent du catalogue.`);
    const node = render.call(this, component, surface, basePath);
    if (node instanceof Element) {
      node.dataset.componentId = id;
      node.dataset.component = type;
    }
    return node;
  }

  #child(component, surface, basePath, prop = "child") {
    const id = this.#childId(component, surface, basePath, prop);
    if (!id) return this.#placeholder("Enfant manquant.");
    return this.#renderComponent(id, surface, basePath);
  }

  #text(value) {
    return value === null || value === undefined ? "" : String(value);
  }

  #placeholder(message) {
    const node = document.createElement("div");
    node.className = "a2ui-placeholder";
    node.textContent = message;
    return node;
  }

  // ── Catalogue : composants de base ───────────────────────────────────────

  get #registry() {
    const self = this;
    return {
      Text(component, surface, basePath) {
        const value = self.#text(self.resolve(component.text, surface, basePath));
        const node = document.createElement("div");
        node.className = `a2ui-text ${component.variant === "caption" ? "is-caption" : ""}`;
        node.append(...formatInlineMarkdown(value));
        return node;
      },

      Row(component, surface, basePath) {
        const node = document.createElement("div");
        node.className = "a2ui-row";
        if (component.justify) node.dataset.justify = component.justify;
        if (component.align) node.dataset.align = component.align;
        for (const child of self.#children(component, surface, basePath)) {
          const element = self.#renderComponent(child.id, surface, child.basePath);
          const childComponent = surface.get(child.id);
          if (childComponent?.weight) element.style.flexGrow = String(childComponent.weight);
          node.append(element);
        }
        return node;
      },

      Column(component, surface, basePath) {
        const node = document.createElement("div");
        node.className = "a2ui-column";
        if (component.justify) node.dataset.justify = component.justify;
        if (component.align) node.dataset.align = component.align;
        for (const child of self.#children(component, surface, basePath)) {
          node.append(self.#renderComponent(child.id, surface, child.basePath));
        }
        return node;
      },

      List(component, surface, basePath) {
        const node = document.createElement("div");
        node.className = `a2ui-list is-${component.direction || "vertical"}`;
        for (const child of self.#children(component, surface, basePath)) {
          node.append(self.#renderComponent(child.id, surface, child.basePath));
        }
        return node;
      },

      Card(component, surface, basePath) {
        const node = document.createElement("section");
        node.className = "a2ui-card";
        node.append(self.#child(component, surface, basePath));
        return node;
      },

      Divider(component) {
        const node = document.createElement("hr");
        node.className = `a2ui-divider ${component.axis || "horizontal"}`;
        return node;
      },

      Button(component, surface, basePath) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `a2ui-button is-${component.variant || "default"}`;
        button.append(self.#child(component, surface, basePath));
        const action = component.action;
        if (action?.event) {
          button.addEventListener("click", () => {
            self.#dispatch(surface, action.event, basePath);
          });
        }
        return button;
      },

      // ── Catalogue personnalisé LPO ─────────────────────────────────────
      KpiCard(component, surface, basePath) {
        const tone = self.#text(self.resolve(component.tone, surface, basePath)) || "neutral";
        const node = document.createElement("div");
        node.className = `a2ui-kpi is-${tone}`;
        const label = document.createElement("div");
        label.className = "kpi-label";
        label.textContent = self.#text(self.resolve(component.label, surface, basePath));
        const value = document.createElement("div");
        value.className = "kpi-value";
        value.textContent = self.#text(self.resolve(component.value, surface, basePath));
        node.append(label, value);
        for (const [prop, className] of [
          ["hint", "kpi-hint"],
          ["trend", "kpi-trend"],
        ]) {
          const text = self.#text(self.resolve(component[prop], surface, basePath));
          if (!text) continue;
          const extra = document.createElement("div");
          extra.className = className;
          extra.textContent = text;
          node.append(extra);
        }
        return node;
      },

      LineChart(component, surface, basePath) {
        const series = self.resolve(component.series, surface, basePath);
        const title = self.#text(self.resolve(component.title, surface, basePath));
        const emptyMessage = self.#text(self.resolve(component.emptyMessage, surface, basePath));
        const node = document.createElement("figure");
        node.className = "a2ui-chart";
        node.append(caption(title));
        if (!Array.isArray(series) || series.length === 0) {
          node.append(self.#placeholder(emptyMessage || "Aucune série à afficher."));
          return node;
        }
        node.append(drawLineChart(series, {
          height: Number(component.height) || 320,
          xLabel: self.#text(self.resolve(component.xLabel, surface, basePath)),
          yLabel: self.#text(self.resolve(component.yLabel, surface, basePath)),
        }));
        return node;
      },

      BarChart(component, surface, basePath) {
        const items = self.resolve(component.items, surface, basePath);
        const title = self.#text(self.resolve(component.title, surface, basePath));
        const emptyMessage = self.#text(self.resolve(component.emptyMessage, surface, basePath));
        const node = document.createElement("figure");
        node.className = "a2ui-chart";
        node.append(caption(title));
        if (!Array.isArray(items) || items.length === 0) {
          node.append(self.#placeholder(emptyMessage || "Aucun classement à afficher."));
          return node;
        }
        node.append(drawBarChart(items.slice(0, Number(component.maxItems) || items.length)));
        return node;
      },

      MapChart(component, surface, basePath) {
        const points = self.resolve(component.points, surface, basePath);
        const title = self.#text(self.resolve(component.title, surface, basePath));
        const emptyMessage = self.#text(self.resolve(component.emptyMessage, surface, basePath));
        const privacy = self.#text(self.resolve(component.privacyMode, surface, basePath)) || "exact";
        const node = document.createElement("figure");
        node.className = "a2ui-chart a2ui-map";
        node.append(caption(title));
        if (!Array.isArray(points) || points.length === 0) {
          node.append(self.#placeholder(emptyMessage || "Aucun point géolocalisé."));
          return node;
        }
        const figure = document.createElement("div");
        figure.className = "map-host";
        node.append(figure);
        const badge = document.createElement("div");
        badge.className = `privacy-badge is-${privacy}`;
        badge.textContent =
          privacy === "exact"
            ? "Localisation : coordonnées exactes"
            : privacy === "blurred"
              ? "Localisation : maille agrégée (espèces sensibles)"
              : "Localisation : échelle départementale";
        node.append(badge);
        this.loadGeo().then((geo) => {
          figure.replaceChildren(drawMap(points, { geo, height: Number(component.height) || 460 }));
        });
        return node;
      },

      DataTable(component, surface, basePath) {
        const rows = self.resolve(component.rows, surface, basePath);
        const title = self.#text(self.resolve(component.title, surface, basePath));
        const captionText = self.#text(self.resolve(component.caption, surface, basePath));
        const emptyMessage = self.#text(self.resolve(component.emptyMessage, surface, basePath));
        const node = document.createElement("figure");
        node.className = "a2ui-table";
        if (title) node.append(caption(title));
        if (!Array.isArray(rows) || rows.length === 0) {
          node.append(self.#placeholder(emptyMessage || "Aucune ligne."));
          return node;
        }
        const columns = Array.isArray(component.columns) ? component.columns : [];
        node.append(drawTable(columns, rows, Number(component.maxRows) || rows.length));
        if (captionText) {
          const note = document.createElement("figcaption");
          note.className = "table-caption";
          note.textContent = captionText;
          node.append(note);
        }
        return node;
      },

      Notice(component, surface, basePath) {
        const tone = self.#text(self.resolve(component.tone, surface, basePath)) || "info";
        const node = document.createElement("div");
        node.className = `a2ui-notice is-${tone}`;
        node.setAttribute("role", tone === "error" ? "alert" : "status");
        const title = self.#text(self.resolve(component.title, surface, basePath));
        if (title) {
          const heading = document.createElement("strong");
          heading.className = "notice-title";
          heading.textContent = title;
          node.append(heading);
        }
        const message = document.createElement("span");
        message.className = "notice-message";
        message.textContent = self.#text(self.resolve(component.message, surface, basePath));
        node.append(message);
        return node;
      },

      SelectField(component, surface, basePath) {
        const label = self.#text(self.resolve(component.label, surface, basePath));
        const current = self.#text(self.resolve(component.value, surface, basePath));
        const options = self.resolve(component.options, surface, basePath);
        const wrapper = document.createElement("label");
        wrapper.className = "a2ui-select";
        const captionEl = document.createElement("span");
        captionEl.textContent = label;
        wrapper.append(captionEl);
        const select = document.createElement("select");
        for (const option of Array.isArray(options) ? options : []) {
          const opt = document.createElement("option");
          opt.value = String(option.value ?? "");
          opt.textContent = self.#text(self.resolve(option.label, surface, "") ?? option.value ?? "");
          if (opt.value === current) opt.selected = true;
          select.append(opt);
        }
        select.addEventListener("change", () => {
          // 1) le contrôle lié met à jour le data model localement…
          if (self.#isBinding(component.value)) {
            const path = component.value["@path"];
            const target = path.startsWith("/") ? path : `${basePath}/${path}`;
            surface.data.set(target, select.value);
          }
          // 2) …puis l'action repart vers l'agent, qui rappellera les outils MCP.
          if (component.action?.event) self.#dispatch(surface, component.action.event, basePath);
        });
        wrapper.append(select);
        return wrapper;
      },
    };
  }

  // ── Actions et fond de carte ─────────────────────────────────────────────

  #dispatch(surface, event, basePath) {
    const name = event?.name;
    if (!name) return;
    const context = {};
    for (const [key, value] of Object.entries(event.context || {})) {
      context[key] = this.resolve(value, surface, basePath);
    }
    const payload = {
      name,
      surfaceId: surface.surfaceId,
      context,
      userMessage: this.#text(this.resolve(event.userMessage, surface, basePath)) || undefined,
    };
    // `sendDataModel: true` : l'agent reçoit l'état complet de la surface.
    if (surface.sendDataModel) payload.dataModel = surface.data.root;
    if (this.onAction) this.onAction(payload);
  }

  loadGeo() {
    if (this.geo) return Promise.resolve(this.geo);
    if (this.geoPromise) return this.geoPromise;
    this.geoPromise = fetch("/static/geo/departements.geojson")
      .then((response) => (response.ok ? response.json() : null))
      .then((geo) => {
        this.geo = geo;
        return geo;
      })
      .catch(() => null);
    return this.geoPromise;
  }
}

// ── Primitives de rendu (SVG / DOM) ─────────────────────────────────────────

function caption(text) {
  const node = document.createElement("figcaption");
  node.className = "chart-title";
  node.textContent = text || "";
  return node;
}

/** Markdown minimal **gras** / *italique*, sans injection HTML. */
function formatInlineMarkdown(text) {
  const parts = [];
  const pattern = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let lastIndex = 0;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) parts.push(document.createTextNode(text.slice(lastIndex, match.index)));
    const token = match[0];
    const element = document.createElement(token.startsWith("**") ? "strong" : "em");
    element.textContent = token.replace(/\*\*/g, "").replace(/\*/g, "");
    parts.push(element);
    lastIndex = match.index + token.length;
  }
  if (lastIndex < text.length) parts.push(document.createTextNode(text.slice(lastIndex)));
  return parts.length ? parts : [document.createTextNode("")];
}

function svgEl(name, attrs = {}) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== null && value !== undefined) node.setAttribute(key, String(value));
  }
  return node;
}

function niceNumber(value) {
  if (value <= 0) return 1;
  const exponent = Math.floor(Math.log10(value));
  const fraction = value / 10 ** exponent;
  const nice = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10;
  return nice * 10 ** exponent;
}

function formatValue(value, decimals = 0) {
  const number = Number(value);
  if (!Number.isFinite(number)) return String(value ?? "");
  return number.toLocaleString("fr-FR", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

/** Courbes multi-séries, axes et légende, en SVG pur. */
function drawLineChart(series, { height = 320, xLabel = "", yLabel = "" } = {}) {
  const width = 720;
  const margin = { top: 18, right: 18, bottom: 46, left: 56 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  const labels = [];
  for (const one of series) {
    for (const point of one.points || []) {
      const key = String(point.x);
      if (!labels.includes(key)) labels.push(key);
    }
  }
  labels.sort();
  const xIndex = new Map(labels.map((label, index) => [label, index]));
  const maxY = Math.max(
    1,
    ...series.flatMap((one) => (one.points || []).map((point) => Number(point.y) || 0)),
  );
  const step = niceNumber(maxY / 4);
  const yMax = Math.ceil(maxY / step) * step;

  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    class: "chart-svg",
    role: "img",
    "aria-label": `Graphique linéaire : ${series.map((s) => s.label).join(", ")}`,
  });

  const x = (label) => margin.left + (plotWidth * (xIndex.get(label) ?? 0)) / Math.max(1, labels.length - 1 || 1);
  const y = (value) => margin.top + plotHeight - (plotHeight * Number(value || 0)) / yMax;

  // Grille + graduations Y
  for (let value = 0; value <= yMax + 0.001; value += step) {
    svg.append(
      svgEl("line", {
        x1: margin.left,
        x2: margin.left + plotWidth,
        y1: y(value),
        y2: y(value),
        class: "grid-line",
      }),
    );
    const tick = svgEl("text", {
      x: margin.left - 8,
      y: y(value) + 4,
      class: "axis-label",
      "text-anchor": "end",
    });
    tick.textContent = formatValue(value);
    svg.append(tick);
  }

  // Graduations X (max ~10 libellés)
  const stride = Math.max(1, Math.ceil(labels.length / 10));
  labels.forEach((label, index) => {
    if (index % stride !== 0) return;
    const tick = svgEl("text", {
      x: x(label),
      y: margin.top + plotHeight + 20,
      class: "axis-label",
      "text-anchor": "middle",
    });
    tick.textContent = label;
    svg.append(tick);
  });

  // Séries
  const palette = ["#1f4e79", "#2e7d32", "#b26a00", "#7b1fa2", "#00695c", "#c62828"];
  series.forEach((one, index) => {
    const color = one.color || palette[index % palette.length];
    const points = (one.points || []).filter((point) => xIndex.has(String(point.x)));
    if (!points.length) return;
    const path = points
      .map((point, i) => `${i === 0 ? "M" : "L"}${x(String(point.x))},${y(point.y)}`)
      .join(" ");
    svg.append(svgEl("path", { d: path, fill: "none", stroke: color, "stroke-width": 2.5, "stroke-linejoin": "round" }));
    points.forEach((point) => {
      const circle = svgEl("circle", { cx: x(String(point.x)), cy: y(point.y), r: 3.2, fill: color });
      const title = svgEl("title");
      title.textContent = `${one.label} · ${point.x} : ${formatValue(point.y)}`;
      circle.append(title);
      svg.append(circle);
    });
    if (series.length > 1) {
      const legendY = margin.top + 6 + index * 18;
      svg.append(svgEl("rect", { x: width - margin.right - 150, y: legendY - 9, width: 10, height: 10, fill: color, rx: 2 }));
      const text = svgEl("text", { x: width - margin.right - 134, y: legendY, class: "legend-label" });
      text.textContent = one.label;
      svg.append(text);
    }
  });

  if (yLabel) {
    const label = svgEl("text", {
      x: 14,
      y: margin.top + plotHeight / 2,
      class: "axis-title",
      transform: `rotate(-90 14 ${margin.top + plotHeight / 2})`,
      "text-anchor": "middle",
    });
    label.textContent = yLabel;
    svg.append(label);
  }
  if (xLabel) {
    const label = svgEl("text", {
      x: margin.left + plotWidth / 2,
      y: height - 8,
      class: "axis-title",
      "text-anchor": "middle",
    });
    label.textContent = xLabel;
    svg.append(label);
  }
  return svg;
}

/** Barres horizontales (libellés longs d'espèces lisibles sans rotation). */
function drawBarChart(items) {
  const max = Math.max(1, ...items.map((item) => Number(item.value) || 0));
  const list = document.createElement("div");
  list.className = "bar-list";
  for (const item of items) {
    const row = document.createElement("div");
    row.className = "bar-row";

    const label = document.createElement("div");
    label.className = "bar-label";
    label.textContent = String(item.label ?? "");

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${Math.max(2, ((Number(item.value) || 0) / max) * 100)}%`;
    track.append(fill);

    const value = document.createElement("div");
    value.className = "bar-value";
    value.textContent = formatValue(item.value);

    row.append(label, track, value);
    if (item.hint) {
      const hint = document.createElement("div");
      hint.className = "bar-hint";
      hint.textContent = String(item.hint);
      row.append(hint);
    }
    row.title = `${item.label} : ${formatValue(item.value)}`;
    list.append(row);
  }
  return list;
}

/** Carte : fond départemental (GeoJSON) + points projetés. */
function drawMap(points, { geo = null, height = 460 } = {}) {
  const width = 720;
  const bounds = { lonMin: -5.4, lonMax: 9.8, latMin: 41.2, latMax: 51.3 };
  const project = (lon, lat) => {
    const scaleX = width / (bounds.lonMax - bounds.lonMin);
    const scaleY = height / (bounds.latMax - bounds.latMin);
    const x = (lon - bounds.lonMin) * scaleX;
    const y = (bounds.latMax - lat) * scaleY;
    return [x, y];
  };

  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    class: "map-svg",
    role: "img",
    "aria-label": `Carte de ${points.length} mailles d'observation`,
  });

  svg.append(svgEl("rect", { x: 0, y: 0, width, height, class: "map-bg" }));

  if (geo?.features) {
    for (const feature of geo.features) {
      const polygons = feature.geometry?.coordinates || [];
      const path = polygons
        .map((rings) =>
          rings
            .map((ring) =>
              ring
                .map(([lon, lat], index) => {
                  const [x, y] = project(lon, lat);
                  return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
                })
                .join(" ") + "Z",
            )
            .join(" "),
        )
        .join(" ");
      if (!path) continue;
      const outline = svgEl("path", { d: path, class: "map-dept", "fill-rule": "evenodd" });
      const title = svgEl("title");
      title.textContent = feature.properties?.nom || "";
      outline.append(title);
      svg.append(outline);
    }
  }

  const maxValue = Math.max(1, ...points.map((point) => Number(point.value) || 0));
  for (const point of points) {
    const [x, y] = project(Number(point.lon), Number(point.lat));
    const radius = 3 + (12 * Math.sqrt((Number(point.value) || 0) / maxValue));
    const circle = svgEl("circle", {
      cx: x,
      cy: y,
      r: radius,
      class: point.sensitive ? "map-point is-sensitive" : "map-point",
    });
    const title = svgEl("title");
    title.textContent = `${point.label || "maille"} · ${formatValue(point.value)} observation(s)${
      point.sensitive ? " · espèce sensible (localisation floutée)" : ""
    }`;
    circle.append(title);
    svg.append(circle);
  }
  return svg;
}

/** Tableau typé (colonnes déclaratives, valeurs venant du data model). */
function drawTable(columns, rows, maxRows) {
  const table = document.createElement("table");
  table.className = "data-table";
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const column of columns) {
    const th = document.createElement("th");
    th.textContent = column.label || column.key;
    if (column.align) th.style.textAlign = column.align;
    headRow.append(th);
  }
  thead.append(headRow);
  table.append(thead);

  const tbody = document.createElement("tbody");
  for (const row of rows.slice(0, maxRows)) {
    const tr = document.createElement("tr");
    for (const column of columns) {
      const td = document.createElement("td");
      td.textContent = formatCell(row[column.key], column.format);
      if (column.align) td.style.textAlign = column.align;
      tr.append(td);
    }
    tbody.append(tr);
  }
  table.append(tbody);
  return table;
}

function formatCell(value, format) {
  if (value === null || value === undefined) return "—";
  switch (format) {
    case "integer":
      return formatValue(value);
    case "decimal":
      return formatValue(value, 1);
    case "percent":
      return `${formatValue(Number(value) * 100, 1)} %`;
    case "date":
      return String(value).slice(0, 10);
    default:
      return String(value);
  }
}
