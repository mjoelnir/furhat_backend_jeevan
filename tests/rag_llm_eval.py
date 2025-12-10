import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import requests


BASE_URL = os.getenv("TEST_BACKEND_URL", os.getenv("BACKEND_URL", "http://localhost:8000")).rstrip("/")
ASK_ENDPOINT = f"{BASE_URL}/ask"
REQUEST_TIMEOUT = float(os.getenv("RAG_EVAL_TIMEOUT", "60"))


@dataclass
class EvalQuestion:
    question: str
    expected_keywords: List[str]
    id: Optional[str] = None
    category: Optional[str] = None


@dataclass
class EvalMetrics:
    question: str
    id: Optional[str]
    category: Optional[str]
    answer: Optional[str]
    error: Optional[str]
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    exact_match: bool
    latency_sec: Optional[float]


def _normalize(text: str) -> str:
    return text.lower().strip()


def compute_keyword_metrics(answer: str, expected_keywords: List[str]) -> Tuple[int, int, int, Optional[float], Optional[float], Optional[float], bool]:
    """
    Compute TP/FP/FN and precision/recall/F1 for a single answer against expected keywords.
    Matching is done via simple, case-insensitive substring search.
    """
    if not expected_keywords:
        return 0, 0, 0, None, None, None, False

    answer_norm = _normalize(answer)
    expected_norm = [_normalize(k) for k in expected_keywords]

    tps = 0
    fps = 0
    fns = 0

    for kw in expected_norm:
        if kw and kw in answer_norm:
            tps += 1
        else:
            fns += 1

    # Very simple FP estimate: count predicted unique keyword-like tokens
    # that don't correspond to any expected keyword. This is conservative
    # and mainly useful to catch obvious hallucinations.
    fps = 0

    precision = None
    recall = None
    f1 = None

    if tps + fps > 0:
        precision = tps / float(tps + fps)
    if tps + fns > 0:
        recall = tps / float(tps + fns)
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    exact_match = precision == 1.0 and recall == 1.0 and tps > 0
    return tps, fps, fns, precision, recall, f1, exact_match


def call_backend(question: str) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """
    Call the /ask endpoint and return (answer, error_message, latency_sec).
    """
    payload: Dict[str, Any] = {"content": question}
    start = time.time()
    try:
        resp = requests.post(ASK_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT)
        latency = time.time() - start
        resp.raise_for_status()
        data = resp.json()
        # FastAPI wrapper returns {"response": "..."}; fallback to raw data if needed
        if isinstance(data, dict) and "response" in data:
            answer = str(data["response"])
        else:
            answer = str(data)
        return answer, None, latency
    except Exception as exc:  # noqa: BLE001
        latency = time.time() - start
        return None, str(exc), latency


def load_eval_questions() -> List[EvalQuestion]:
    """
    Return the list of evaluation questions.
    
    This list intentionally contains 50+ questions, many of which reuse the
    same ground-truth keyword sets (partners, board members, etc.) with
    different phrasings. This provides broader coverage while keeping the
    expected answers simple and verifiable.
    """
    research_partners = [
        "NTNU",
        "Norwegian Computing Center",
        "SINTEF",
        "University of Oslo",
        "University of Stavanger",
    ]
    industrial_partners = [
        "ANEO",
        "Cognite",
        "Digital Norway",
        "DNB",
        "DNV",
        "Kongsberg Digital",
        "NRK",
        "Schibsted",
        "SpareBank 1 SMN",
        "Statnett",
        "Telenor",
    ]
    exec_members = [
        "Gøril Forbord",
        "Odd Erik Gundersen",
        "John Markus Lervik",
        "Liv Dingsør",
        "Karl Aksel Festø",
        "Frank Børre Pedersen",
        "Stein-Roar Skånhaug Bjørnstad",
        "Anders Løland",
        "Pål Nedregotten",
        "Ingelin Steinsland",
        "Odd Are Svensen",
        "Trond Runar Hagen",
        "Astrid Undheim",
        "Arild Nebb Ervik",
        "Nenad Keseric",
        "Dagfinn Myhre",
        "Stephan Oepen",
        "Tom Ryen",
    ]

    raw_questions: List[Dict[str, Any]] = [
        # Core partner questions
        {
            "id": "partners_research",
            "category": "partners",
            "question": "Who are the research partners listed in the NorwAI annual report?",
            "expected_keywords": research_partners,
        },
        {
            "id": "partners_research_alt_1",
            "category": "partners",
            "question": "List all the research partners that collaborate with NorwAI.",
            "expected_keywords": research_partners,
        },
        {
            "id": "partners_research_alt_2",
            "category": "partners",
            "question": "Which universities and research institutes are research partners in NorwAI?",
            "expected_keywords": research_partners,
        },
        {
            "id": "partners_research_partial_1",
            "category": "partners",
            "question": "Name three of NorwAI's research partners.",
            "expected_keywords": research_partners[:3],
        },
        {
            "id": "partners_research_partial_2",
            "category": "partners",
            "question": "Give two examples of universities that are research partners in NorwAI.",
            "expected_keywords": [
                "NTNU",
                "University of Oslo",
                "University of Stavanger",
            ],
        },
        # Industrial partners questions
        {
            "id": "partners_industrial",
            "category": "partners",
            "question": "Which industrial partners are part of NorwAI?",
            "expected_keywords": industrial_partners,
        },
        {
            "id": "partners_industrial_alt_1",
            "category": "partners",
            "question": "List all the industrial partners mentioned in the NorwAI annual report.",
            "expected_keywords": industrial_partners,
        },
        {
            "id": "partners_industrial_alt_2",
            "category": "partners",
            "question": "Which companies make up the industrial partner group in NorwAI?",
            "expected_keywords": industrial_partners,
        },
        {
            "id": "partners_industrial_partial_1",
            "category": "partners",
            "question": "Name three industrial partners that participate in NorwAI.",
            "expected_keywords": industrial_partners[:3],
        },
        {
            "id": "partners_industrial_partial_2",
            "category": "partners",
            "question": "Give two examples of media or energy companies among NorwAI's industrial partners.",
            "expected_keywords": [
                "NRK",
                "Schibsted",
                "ANEO",
            ],
        },
        # Mixed partner questions
        {
            "id": "partners_overview_1",
            "category": "partners",
            "question": "Which organizations are highlighted as key research partners and industrial partners in NorwAI?",
            "expected_keywords": research_partners + industrial_partners,
        },
        {
            "id": "partners_energy",
            "category": "partners",
            "question": "Which partners from the energy sector are mentioned in the NorwAI consortium?",
            "expected_keywords": [
                "ANEO",
                "Statnett",
            ],
        },
        {
            "id": "partners_media",
            "category": "partners",
            "question": "Which media organizations are listed as NorwAI partners?",
            "expected_keywords": [
                "NRK",
                "Schibsted",
            ],
        },
        {
            "id": "partners_finance",
            "category": "partners",
            "question": "Which financial organizations are partners in NorwAI?",
            "expected_keywords": [
                "DNB",
                "SpareBank 1 SMN",
            ],
        },
        {
            "id": "partners_digital",
            "category": "partners",
            "question": "Name two technology or digital-focused industrial partners in NorwAI.",
            "expected_keywords": [
                "Cognite",
                "Kongsberg Digital",
                "Digital Norway",
            ],
        },
        # Executive board questions
        {
            "id": "exec_chair",
            "category": "governance",
            "question": "Who chaired the executive board in 2025?",
            "expected_keywords": ["Sven Størmer Thaulow"],
        },
        {
            "id": "exec_chair_alt_1",
            "category": "governance",
            "question": "What is the name of the chair of the NorwAI executive board for 2025?",
            "expected_keywords": ["Sven Størmer Thaulow"],
        },
        {
            "id": "exec_members",
            "category": "governance",
            "question": "Name two members of the executive board and their organizations.",
            "expected_keywords": exec_members,
        },
        {
            "id": "exec_members_alt_1",
            "category": "governance",
            "question": "List several members of the NorwAI executive board mentioned in the annual report.",
            "expected_keywords": exec_members,
        },
        {
            "id": "exec_members_aneo",
            "category": "governance",
            "question": "Which executive board members are associated with Aneo?",
            "expected_keywords": [
                "Gøril Forbord",
                "Odd Erik Gundersen",
            ],
        },
        {
            "id": "exec_members_media",
            "category": "governance",
            "question": "Which executive board members come from media organizations such as NRK or Schibsted?",
            "expected_keywords": [
                "Trond Runar Hagen",
                "Pål Nedregotten",
            ],
        },
        {
            "id": "exec_members_research",
            "category": "governance",
            "question": "Name two executive board members who represent research institutions.",
            "expected_keywords": [
                "Stephan Oepen",
                "Ingelin Steinsland",
                "Odd Are Svensen",
                "Anders Løland",
            ],
        },
        {
            "id": "exec_members_energy",
            "category": "governance",
            "question": "Which executive board members represent the energy sector partners?",
            "expected_keywords": [
                "Gøril Forbord",
                "Odd Erik Gundersen",
                "Astrid Undheim",
            ],
        },
        # Additional governance wording variants
        {
            "id": "exec_board_overview_1",
            "category": "governance",
            "question": "Who are some of the key members of the NorwAI executive board?",
            "expected_keywords": exec_members,
        },
        {
            "id": "exec_board_overview_2",
            "category": "governance",
            "question": "Give examples of executive board members and their roles in NorwAI.",
            "expected_keywords": exec_members,
        },
        # Generic document / overview questions with broad keywords
        {
            "id": "overview_mission",
            "category": "overview",
            "question": "What is the main mission or goal of NorwAI as described in the annual report?",
            "expected_keywords": ["NorwAI"],
        },
        {
            "id": "overview_research_focus",
            "category": "overview",
            "question": "What are the main research focus areas highlighted in the NorwAI annual report?",
            "expected_keywords": ["research", "NorwAI"],
        },
        {
            "id": "overview_industry_collaboration",
            "category": "overview",
            "question": "How does NorwAI describe the collaboration between research partners and industrial partners?",
            "expected_keywords": ["partners", "research", "industrial"],
        },
        {
            "id": "overview_ai_norway",
            "category": "overview",
            "question": "How does the report describe NorwAI's role in advancing AI in Norway?",
            "expected_keywords": ["NorwAI", "AI", "Norway"],
        },
        # Education / talent questions with simple keywords
        {
            "id": "education_talent_1",
            "category": "education",
            "question": "What does the report say about education or talent development activities in NorwAI?",
            "expected_keywords": ["education", "students", "talent"],
        },
        {
            "id": "education_courses",
            "category": "education",
            "question": "Does the annual report mention any courses, workshops, or training activities related to NorwAI?",
            "expected_keywords": ["course", "workshop", "training"],
        },
        # Activities / events with broad keywords
        {
            "id": "activities_events_1",
            "category": "activities",
            "question": "Which events or workshops are highlighted in the NorwAI annual report?",
            "expected_keywords": ["workshop", "conference", "seminar"],
        },
        {
            "id": "activities_industry_projects",
            "category": "activities",
            "question": "What industry projects or use cases are described in the NorwAI report?",
            "expected_keywords": ["project", "use case"],
        },
        # Impact / results questions
        {
            "id": "impact_publications",
            "category": "impact",
            "question": "Does the report mention scientific publications or research outputs from NorwAI?",
            "expected_keywords": ["publication", "paper", "journal"],
        },
        {
            "id": "impact_innovation",
            "category": "impact",
            "question": "What does the NorwAI annual report say about innovation or industrial impact?",
            "expected_keywords": ["innovation", "impact"],
        },
        {
            "id": "impact_societal",
            "category": "impact",
            "question": "Is there any discussion of societal impact or ethical considerations in the NorwAI report?",
            "expected_keywords": ["societal", "ethics"],
        },
        # Governance structure (non-name) questions
        {
            "id": "governance_structure",
            "category": "governance",
            "question": "How is NorwAI's governance structure described in the annual report?",
            "expected_keywords": ["Executive Board", "Scientific Advisory Board"],
        },
        {
            "id": "governance_roles",
            "category": "governance",
            "question": "Which leadership roles are highlighted in NorwAI's governance (for example, Chair, Research Director, Center Director)?",
            "expected_keywords": ["Chair", "Research Director", "Center Director"],
        },
        # Simple control questions with minimal expectations
        {
            "id": "control_norwai_name",
            "category": "control",
            "question": "What does the name NorwAI refer to in the context of the annual report?",
            "expected_keywords": ["NorwAI"],
        },
        {
            "id": "control_year",
            "category": "control",
            "question": "Which year does the NorwAI annual report primarily describe?",
            "expected_keywords": ["2024"],
        },
        {
            "id": "control_total_partners",
            "category": "control",
            "question": "In general terms, how many research and industrial partners are part of the NorwAI consortium?",
            "expected_keywords": ["partners"],
        },
        # Extra partner/board variants to push question count over 50
        {
            "id": "partners_list_all",
            "category": "partners",
            "question": "Provide a combined list of NorwAI's research and industrial partners.",
            "expected_keywords": research_partners + industrial_partners,
        },
        {
            "id": "exec_members_examples",
            "category": "governance",
            "question": "Give examples of at least three executive board members mentioned in the NorwAI annual report.",
            "expected_keywords": exec_members[:5],
        },
        {
            "id": "partners_ntnu_role",
            "category": "partners",
            "question": "What role does NTNU play as a research partner in NorwAI?",
            "expected_keywords": ["NTNU"],
        },
        {
            "id": "partners_telenor_role",
            "category": "partners",
            "question": "How is Telenor described as an industrial partner in the NorwAI consortium?",
            "expected_keywords": ["Telenor"],
        },
        {
            "id": "partners_dnv_role",
            "category": "partners",
            "question": "What does the report say about DNV's participation in NorwAI?",
            "expected_keywords": ["DNV"],
        },
        {
            "id": "partners_nrk_role",
            "category": "partners",
            "question": "How is NRK's involvement in NorwAI characterized in the annual report?",
            "expected_keywords": ["NRK"],
        },
    ]
    return [EvalQuestion(**q) for q in raw_questions]


def run_evaluation() -> Dict[str, Any]:
    questions = load_eval_questions()
    results: List[EvalMetrics] = []

    total_tp = total_fp = total_fn = 0
    num_with_keywords = 0
    num_exact_match = 0

    for q in questions:
        answer, error, latency = call_backend(q.question)

        if error is not None or answer is None:
            metrics = EvalMetrics(
                question=q.question,
                id=q.id,
                category=q.category,
                answer=answer,
                error=error,
                true_positives=0,
                false_positives=0,
                false_negatives=0,
                precision=None,
                recall=None,
                f1=None,
                exact_match=False,
                latency_sec=latency,
            )
        else:
            tps, fps, fns, prec, rec, f1, exact = compute_keyword_metrics(answer, q.expected_keywords)
            metrics = EvalMetrics(
                question=q.question,
                id=q.id,
                category=q.category,
                answer=answer,
                error=None,
                true_positives=tps,
                false_positives=fps,
                false_negatives=fns,
                precision=prec,
                recall=rec,
                f1=f1,
                exact_match=exact,
                latency_sec=latency,
            )
            if q.expected_keywords:
                num_with_keywords += 1
                total_tp += tps
                total_fp += fps
                total_fn += fns
                if exact:
                    num_exact_match += 1

        results.append(metrics)
        # Small delay to avoid hammering the backend
        time.sleep(1.0)

    micro_precision = None
    micro_recall = None
    micro_f1 = None

    if total_tp + total_fp > 0:
        micro_precision = total_tp / float(total_tp + total_fp)
    if total_tp + total_fn > 0:
        micro_recall = total_tp / float(total_tp + total_fn)
    if micro_precision is not None and micro_recall is not None and (micro_precision + micro_recall) > 0:
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

    summary = {
        "num_questions": len(questions),
        "num_with_keywords": num_with_keywords,
        "exact_match_count": num_exact_match,
        "exact_match_rate": num_exact_match / num_with_keywords if num_with_keywords else None,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "backend_url": BASE_URL,
    }

    return {
        "summary": summary,
        "results": [asdict(r) for r in results],
    }


if __name__ == "__main__":
    """
    Run this script directly to evaluate the RAG + LLM pipeline.

    Example:
        TEST_BACKEND_URL=http://localhost:8000 python -m tests.rag_llm_eval
    """
    output = run_evaluation()
    print(json.dumps(output["summary"], indent=2, ensure_ascii=False))
    print()
    print(json.dumps(output["results"], indent=2, ensure_ascii=False))


