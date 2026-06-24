"""
Assistant SQL — interface chat avec traces EN DIRECT.

Les étapes de l'agent (exploration du schéma, écriture/validation/exécution du
SQL, auto-corrections) s'affichent en temps réel dans un `st.status` pendant le
traitement (streaming `stream_mode="updates"`), puis se replient. La réponse, le
graphique et les données restent mis en avant. À la relecture de l'historique,
les étapes sont consultables dans un expander repliable.

Flux : question → agent deepagents (streaming) → ré-exécution lecture seule du
SQL final → cleaning → visualisation déterministe.
"""

from __future__ import annotations

import logging

import streamlit as st

from ...ai import agent as deep_agent
from ...ai.cleaning import clean_dataframe
from ...ai.model_factory import available_providers
from ...ai.viz_engine import create_figure, infer_chart_type
from ...core.models import ChartType
from ...database import engine

logger = logging.getLogger(__name__)

EXAMPLE_QUESTIONS = [
    "Combien y a-t-il d'enregistrements par catégorie ?",
    "Quelles sont les 10 régions avec le plus de zones sensibles ?",
    "Évolution mensuelle des créations",
    "Quelles structures gèrent des zones liées aux espèces ?",
    "Montre un aperçu des tables disponibles",
    "Top 5 des valeurs les plus fréquentes",
]

_TOOL_ICONS = {
    "list_tables": "📚",
    "describe_tables": "🔎",
    "check_sql": "✅",
    "run_sql_query": "🗄️",
    "write_todos": "📝",
    "read_file": "📄",
    "ls": "📁",
    "task": "🤝",
}


def _step_line(tool: str, args: dict, status: str) -> str:
    icon = _TOOL_ICONS.get(tool, "🔧")
    flag = "" if status == "ok" else " ⚠️"
    detail = args.get("tables") or args.get("sql") or ""
    detail = " ".join(str(detail).split())
    if detail:
        detail = f" · {detail[:90]}"
    return f"{icon} **{tool}**{flag}{detail}"


def _render_reasoning(steps: list[dict]):
    if not steps:
        return
    with st.expander(f"🧠 Voir le raisonnement ({len(steps)} étape(s))", expanded=False):
        for s in steps:
            st.markdown(_step_line(s["tool"], s["args"], s["status"]))
            st.caption(s["result"])


def _render_result_block(turn: dict, *, show_reasoning: bool = True):
    """Affiche réponse + graphique + données + SQL (+ raisonnement repliable)."""
    if turn.get("answer"):
        st.markdown(turn["answer"])

    df = turn.get("df")
    sql = turn.get("sql")

    if df is not None and not df.empty:
        chart_type = turn.get("chart_type", ChartType.BAR)
        if chart_type == ChartType.TABLE:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            tab_graph, tab_data = st.tabs(["📊 Graphique", "📋 Données"])
            with tab_graph:
                fig = create_figure(df, chart_type, title=turn.get("question", "")[:80])
                st.plotly_chart(fig, use_container_width=True)
            with tab_data:
                st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "📥 Télécharger CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="resultat_lpo.csv",
            mime="text/csv",
            key=f"dl_{turn['id']}",
        )

    if sql:
        with st.expander("🔍 Requête SQL exécutée", expanded=False):
            st.code(sql, language="sql")

    if show_reasoning:
        _render_reasoning(turn.get("steps", []))


class QueryInterfacePage:
    def render(self):
        st.title("🤖 Assistant SQL — LPO")
        st.caption(
            "Pose une question en français. L'agent explore le schéma de la base "
            "(quel qu'il soit, mono ou multi-tables) et génère le SQL en lecture seule."
        )

        if not available_providers():
            st.warning(
                "⚙️ Aucune clé LLM configurée. Renseigne la clé de ton provider "
                "(ex. `ANTHROPIC_API_KEY`) dans le fichier `.env`."
            )

        self._render_examples()
        self._render_chat()

    def _render_examples(self):
        with st.expander("💡 Exemples de questions", expanded=False):
            cols = st.columns(2)
            for i, q in enumerate(EXAMPLE_QUESTIONS):
                if cols[i % 2].button(q, key=f"ex_{i}", use_container_width=True):
                    st.session_state.pending_question = q

    def _render_chat(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Historique : étapes consultables dans l'expander repliable
        for turn in st.session_state.messages:
            with st.chat_message(turn["role"]):
                if turn["role"] == "user":
                    st.markdown(turn["content"])
                else:
                    _render_result_block(turn)

        question = st.session_state.pop("pending_question", None) or st.chat_input(
            "Pose ta question…"
        )
        if not question:
            return

        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            turn = self._answer_live(question)
            # Les étapes ont été montrées en direct → pas de doublon ici
            _render_result_block(turn, show_reasoning=False)
        st.session_state.messages.append(turn)

    def _answer_live(self, question: str) -> dict:
        """Streame les étapes de l'agent en direct dans un st.status."""
        turn: dict = {
            "role": "assistant",
            "id": len(st.session_state.messages),
            "question": question,
        }
        result = None

        with st.status("Analyse en cours…", expanded=True) as status:
            gen = deep_agent.stream_run(question)
            try:
                while True:
                    try:
                        step = next(gen)
                    except StopIteration as stop:
                        result = stop.value
                        break
                    st.write(_step_line(step.tool, step.args, step.status))
            except Exception as e:  # noqa: BLE001  (défensif : stream_run capture déjà ses erreurs)
                logger.error("Erreur streaming agent: %s", e, exc_info=True)
                result = deep_agent.AgentResult(answer="", error=str(e))

            failed = result is None or result.error
            n = len(result.steps) if result else 0
            status.update(
                label="Erreur" if failed else f"Analyse terminée — {n} étape(s)",
                state="error" if failed else "complete",
                expanded=False,
            )

        if result is None:
            turn["answer"] = "❌ Aucun résultat."
            return turn

        turn["steps"] = [vars(s) for s in result.steps]
        if result.error:
            turn["answer"] = f"❌ {result.error}"
            return turn

        turn["answer"] = result.answer
        turn["sql"] = result.sql

        # Ré-exécution lecture seule du SQL final → données complètes + cleaning
        if result.sql:
            try:
                df = clean_dataframe(engine.run_select(result.sql))
                turn["df"] = df
                turn["chart_type"] = infer_chart_type(df)
            except Exception as e:  # noqa: BLE001
                logger.warning("Ré-exécution SQL échouée: %s", e)
                turn["df"] = None

        return turn


query_interface_page = QueryInterfacePage()
