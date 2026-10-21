"""
The Learning Hub - LangGraph Pedagogical Mastery Roadmap.
7-Stage progression covering cyclic state machines, Pregel supersteps, checkpointing, and HITL.
"""

from typing import List, Dict, Any

LANGGRAPH_STAGES: List[Dict[str, Any]] = [
    {
        "stage_number": 1,
        "title": "Core Architecture, Graph Theory & LangGraph Fundamentals",
        "domain_focus": "Directed cyclic graphs vs DAGs, Pregel computation model, Supersteps, and Atomic state boundaries",
        "completed": True,
        "key_topics": ["Pregel Model", "Superstep Synchronization", "StateGraph Primitives", "Linear DAG Failures"]
    },
    {
        "stage_number": 2,
        "title": "State Definition, Reducers & Channels",
        "domain_focus": "TypedDict schemas, Channel reducers (operator.add), Last-write-wins semantics, and Immutable state updates",
        "completed": True,
        "key_topics": ["Annotated Channels", "Merge Reducers", "State Isolation", "Ephemeral Keys"]
    },
    {
        "stage_number": 3,
        "title": "Conditional Routing, Dynamic Loops & Reflection",
        "domain_focus": "Conditional edges, Router functions, Cycle detection, Recursion limits, and Self-reflection agent loops",
        "completed": True,
        "key_topics": ["add_conditional_edges", "Dynamic Dispatch", "Reflection Nodes", "RecursionLimitExceeded"]
    },
    {
        "stage_number": 4,
        "title": "State Persistence, Checkpointing & Time Travel",
        "domain_focus": "MemorySaver, SqliteSaver, Thread IDs, Write-Ahead Logging (WAL), Forking, and Time-travel state replay",
        "completed": True,
        "key_topics": ["Thread Isolation", "Monotonic Versioning", "Checkpointer Protocols", "State Rewind / Branching"]
    },
    {
        "stage_number": 5,
        "title": "Human-in-the-Loop (HITL) & Interactive Approvals",
        "domain_focus": "interrupt_before, interrupt_after, Pausing state machines, External human input injection, and Resuming",
        "completed": True,
        "key_topics": ["GraphInterrupt", "Breakpoint Execution", "Dynamic State Injection", "Approval Workflows"]
    },
    {
        "stage_number": 6,
        "title": "Multi-Agent Architectures, Subgraphs & Deployment",
        "domain_focus": "Supervisor pattern, Hierarchical subgraphs, Swarm collaboration, and Production Pregel serving",
        "completed": True,
        "key_topics": ["Agent Supervisors", "Isolated Subgraph States", "Cross-Graph Communication", "Async Pregel Engine"]
    },
    {
        "stage_number": 7,
        "title": "Staff-Level Interview Prep, Cheatsheet & Appendix",
        "domain_focus": "Staff architecture trade-offs, Distributed consistency, Distributed locking, and Production hardening",
        "completed": True,
        "key_topics": ["Distributed Checkpointing", "Conflict Resolution", "Zero-downtime Migration", "Observability & Tracing"]
    }
]

def get_completed_count() -> int:
    return sum(1 for s in LANGGRAPH_STAGES if s["completed"])

def get_progress_percentage() -> int:
    return round((get_completed_count() / len(LANGGRAPH_STAGES)) * 100)
