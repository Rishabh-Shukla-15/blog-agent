from __future__ import annotations

import operator
import os
import re
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Schemas
# ============================================================

class Task(BaseModel):
    id: int
    title: str
    goal: str
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal[
        "explainer",
        "tutorial",
        "news_roundup",
        "comparison",
        "system_design"
    ] = "explainer"

    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = 5


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class ImageSpec(BaseModel):
    placeholder: str
    filename: str
    alt: str
    caption: str
    prompt: str

    size: Literal[
        "1024x1024",
        "1024x1536",
        "1536x1024"
    ] = "1024x1024"

    quality: Literal[
        "low",
        "medium",
        "high"
    ] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)


class State(TypedDict):
    topic: str

    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    as_of: str
    recency_days: int

    sections: Annotated[List[tuple[int, str]], operator.add]

    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str


Task.model_rebuild()
Plan.model_rebuild()
EvidenceItem.model_rebuild()
RouterDecision.model_rebuild()
EvidencePack.model_rebuild()
ImageSpec.model_rebuild()
GlobalImagePlan.model_rebuild()

# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="llama-3.1-8b-instant"
)

# ============================================================
# Router
# ============================================================

ROUTER_SYSTEM = """
Return ONLY valid JSON.

You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book
- hybrid
- open_book
"""

def router_node(state: State) -> dict:

    decider = llm.with_structured_output(
        RouterDecision,
        method="json_mode"
    )

    try:
        decision = decider.invoke(
            [
                SystemMessage(content=ROUTER_SYSTEM),
                HumanMessage(
                    content=f"""
Topic: {state['topic']}
As-of date: {state['as_of']}
"""
                ),
            ]
        )

    except Exception:

        decision = RouterDecision(
            needs_research=False,
            mode="closed_book",
            reason="Fallback",
            queries=[],
            max_results_per_query=5,
        )

    if decision.mode == "open_book":
        recency_days = 7

    elif decision.mode == "hybrid":
        recency_days = 45

    else:
        recency_days = 3650

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries[:2],
        "recency_days": recency_days,
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"


# ============================================================
# Research
# ============================================================

def _tavily_search(query: str, max_results: int = 2) -> List[dict]:

    if not os.getenv("TAVILY_API_KEY"):
        return []

    try:
        from langchain_community.tools.tavily_search import TavilySearchResults

        tool = TavilySearchResults(max_results=max_results)

        results = tool.invoke({"query": query})

        out = []

        for r in results or []:
            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": r.get("content") or "",
                    "published_at": r.get("published_date"),
                    "source": r.get("source"),
                }
            )

        return out

    except Exception:
        return []


RESEARCH_SYSTEM = """
Return ONLY valid JSON.

You are a research synthesizer.
"""


def research_node(state: State) -> dict:

    queries = (state.get("queries") or [])[:2]

    raw = []

    for q in queries:
        raw.extend(_tavily_search(q, max_results=2))

    if not raw:
        return {"evidence": []}

    extractor = llm.with_structured_output(
        EvidencePack,
        method="json_mode"
    )

    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(content=f"Raw results:\n{raw}")
        ]
    )

    return {"evidence": pack.evidence}


# ============================================================
# Orchestrator
# ============================================================

ORCH_SYSTEM = """
Return ONLY valid JSON.

You are a senior technical writer.

Create 3-4 blog sections only.
"""
def orchestrator_node(state: State) -> dict:

    topic = state["topic"].lower()

    # --------------------------------------------------------
    # Travel Blog
    # --------------------------------------------------------

    if any(x in topic for x in ["trip", "travel", "itinerary", "vacation"]):

        plan = Plan(
            blog_title=state["topic"],
            audience="Travel enthusiasts",
            tone="Helpful and practical",
            blog_kind="explainer",
            constraints=[],
            tasks=[
                Task(
                    id=1,
                    title="Trip Overview",
                    goal="Introduce destination and expectations",
                    bullets=[
                        "Best season",
                        "Visa and flights",
                        "Expected budget"
                    ],
                    target_words=180,
                ),
                Task(
                    id=2,
                    title="12-Day Itinerary",
                    goal="Day-wise travel planning",
                    bullets=[
                        "North Island",
                        "South Island",
                        "Activities and transport"
                    ],
                    target_words=350,
                ),
                Task(
                    id=3,
                    title="Budget Breakdown",
                    goal="Explain medium-budget planning",
                    bullets=[
                        "Hotels",
                        "Food",
                        "Transport and activities"
                    ],
                    target_words=220,
                ),
            ],
        )

    # --------------------------------------------------------
    # Tech Blog
    # --------------------------------------------------------

    else:

        plan = Plan(
            blog_title=f"{state['topic']} Explained",
            audience="Intermediate programmers",
            tone="Educational and concise",
            blog_kind="explainer",
            constraints=[],
            tasks=[
                Task(
                    id=1,
                    title="Introduction",
                    goal="Introduce the concept",
                    bullets=[
                        "What problem it solves",
                        "Why it matters",
                        "Applications"
                    ],
                    target_words=150,
                ),
                Task(
                    id=2,
                    title="Core Idea",
                    goal="Explain main intuition",
                    bullets=[
                        "Key observations",
                        "Algorithm intuition",
                        "Complexity"
                    ],
                    target_words=220,
                ),
                Task(
                    id=3,
                    title="Implementation",
                    goal="Explain implementation details",
                    bullets=[
                        "Data structures",
                        "Code flow",
                        "Optimizations"
                    ],
                    target_words=280,
                    requires_code=True,
                ),
            ],
        )

    return {"plan": plan}

# ============================================================
# Fanout
# ============================================================

def fanout(state: State):

    first_task = state["plan"].tasks[0]

    return [
        Send(
            "worker",
            {
                "task": first_task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [
                    e.model_dump()
                    for e in state.get("evidence", [])
                ],
            },
        )
    ]


# ============================================================
# Worker
# ============================================================

WORKER_SYSTEM = """
Write ONE clean markdown section.
"""

def worker_node(payload: dict) -> dict:

    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])

    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=f"""
Blog title: {plan.blog_title}

Section: {task.title}

Goal:
{task.goal}

Bullets:
{task.bullets}
"""
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}


# ============================================================
# Reducer
# ============================================================

def merge_content(state: State) -> dict:

    plan = state["plan"]

    ordered_sections = [
        md
        for _, md in sorted(
            state["sections"],
            key=lambda x: x[0]
        )
    ]

    body = "\n\n".join(ordered_sections).strip()

    merged_md = f"# {plan.blog_title}\n\n{body}\n"

    return {"merged_md": merged_md}


def decide_images(state: State) -> dict:

    md = state["merged_md"]

    image_specs = [
        {
            "placeholder": "[IMAGE_1]",
            "filename": "cover_image.png",
            "alt": "Blog cover image",
            "caption": "Generated cover image",
            "prompt": f"Professional blog cover image about {state['topic']}"
        }
    ]

    md = (
        f"![Cover Image](images/cover_image.png)\n\n"
        + md
    )

    return {
        "md_with_placeholders": md,
        "image_specs": image_specs,
    }

def generate_and_place_images(state: State) -> dict:

    import requests

    md = state["md_with_placeholders"]

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in state["image_specs"]:

        image_path = images_dir / spec["filename"]

        try:

            prompt = spec["prompt"].replace(" ", "%20")

            url = f"https://image.pollinations.ai/prompt/{prompt}"

            response = requests.get(url, timeout=60)

            if response.status_code == 200:

                with open(image_path, "wb") as f:
                    f.write(response.content)

        except Exception as e:

            print("IMAGE ERROR:", e)

    output_path = outputs_dir / "generated_blog.md"

    output_path.write_text(md, encoding="utf-8")

    return {"final": md}


# ============================================================
# Reducer Subgraph
# ============================================================

reducer_graph = StateGraph(State)

reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node(
    "generate_and_place_images",
    generate_and_place_images
)

reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge(
    "decide_images",
    "generate_and_place_images"
)
reducer_graph.add_edge(
    "generate_and_place_images",
    END
)

reducer_subgraph = reducer_graph.compile()


# ============================================================
# Main Graph
# ============================================================

g = StateGraph(State)

g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")

g.add_conditional_edges(
    "router",
    route_next,
    {
        "research": "research",
        "orchestrator": "orchestrator"
    }
)

g.add_edge("research", "orchestrator")

g.add_conditional_edges(
    "orchestrator",
    fanout,
    ["worker"]
)

g.add_edge("worker", "reducer")

g.add_edge("reducer", END)

app = g.compile()


# ============================================================
# Runner
# ============================================================

def run(topic: str, as_of: Optional[str] = None):

    if as_of is None:
        as_of = date.today().isoformat()

    out = app.invoke(
        {
            "topic": topic,
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "as_of": as_of,
            "recency_days": 7,
            "sections": [],
            "merged_md": "",
            "md_with_placeholders": "",
            "image_specs": [],
            "final": "",
        }
    )

    return out


if __name__ == "__main__":

    out = run("Binary Lifting LCA")

    print(out["final"])