"""
The Learning Hub - LangGraph Production Multi-Agent & Cyclic Orchestrator CLI.
Demonstrates cyclic reflection loops, channel reducers, persistent checkpointing,
time-travel rewind, and Human-in-the-Loop (HITL) approval gates.
"""

from __future__ import annotations
import operator
import sys
from .roadmap import LANGGRAPH_STAGES, get_completed_count, get_progress_percentage
from .graph import (
    START,
    END,
    StateGraph,
    MemorySaver,
    GraphInterrupt,
    AgentState
)


def print_banner() -> None:
    print("=" * 76)
    print("   THE LEARNING HUB: LANGGRAPH CYCLIC STATE MACHINE & AGENT ENGINE   ")
    print("=" * 76)


def display_roadmap() -> None:
    print("\n[+] 7-Stage LangGraph Pedagogical Mastery Roadmap:")
    for stage in LANGGRAPH_STAGES:
        status = "[DONE]" if stage["completed"] else "[TODO]"
        print(f"  {status} Stage {stage['stage_number']}: {stage['title']}")
        print(f"         Focus: {stage['domain_focus']}")
    print(f"\nProgress: {get_completed_count()}/{len(LANGGRAPH_STAGES)} Stages Complete ({get_progress_percentage()}%)\n")


def demonstrate_cyclic_reflection() -> None:
    print("-" * 68)
    print("1. Cyclic Agent State Machine: Self-Correction & Reflection Loop")
    print("-" * 68)

    # State with operator.add channel reducer for message history
    graph = StateGraph(schema={"messages": operator.add})

    def draft_node(state: AgentState) -> AgentState:
        iters = state.get("iterations", 0) + 1
        msg = f"Draft v{iters}: Optimized API architecture plan."
        print(f"    [draft_node] Executing iteration {iters}...")
        return {
            "iterations": iters,
            "messages": [msg],
            "current_draft": msg
        }

    def evaluate_node(state: AgentState) -> AgentState:
        iters = state.get("iterations", 1)
        score = min(1.0, 0.4 + (iters * 0.3))
        feedback = "Refine error recovery semantics." if score < 0.9 else "Production ready!"
        print(f"    [evaluate_node] Quality Score={score:.2f} | Feedback: {feedback}")
        return {
            "quality_score": score,
            "messages": [f"Evaluation: Score {score:.2f} - {feedback}"]
        }

    def should_continue(state: AgentState) -> str:
        score = state.get("quality_score", 0.0)
        iters = state.get("iterations", 0)
        if score >= 0.9 or iters >= 3:
            print("    [conditional_router] -> Target: END (Threshold satisfied)")
            return END
        print("    [conditional_router] -> Target: draft (Revising draft in cyclic loop)")
        return "draft"

    graph.add_node("draft", draft_node)
    graph.add_node("evaluate", evaluate_node)
    graph.set_entry_point("draft")
    graph.add_edge("draft", "evaluate")
    graph.add_conditional_edges("evaluate", should_continue)

    app = graph.compile()
    final_state = app.invoke({"iterations": 0, "messages": ["Initial user request: Design distributed lock."]})
    print(f"\n  Final State Result:")
    print(f"    Total Iterations: {final_state['iterations']}")
    print(f"    Quality Score:    {final_state['quality_score']}")
    print(f"    Total Messages in Channel: {len(final_state['messages'])}\n")


def demonstrate_persistence_and_time_travel() -> None:
    print("-" * 68)
    print("2. Persistent Checkpointing & Time-Travel Rewind (MemorySaver)")
    print("-" * 68)

    saver = MemorySaver()
    graph = StateGraph()

    def step_a(state: AgentState) -> AgentState:
        return {"step": 1, "log": state.get("log", []) + ["Step A completed"]}

    def step_b(state: AgentState) -> AgentState:
        return {"step": 2, "log": state.get("log", []) + ["Step B completed"]}

    def step_c(state: AgentState) -> AgentState:
        return {"step": 3, "log": state.get("log", []) + ["Step C completed"]}

    graph.add_node("step_a", step_a)
    graph.add_node("step_b", step_b)
    graph.add_node("step_c", step_c)
    graph.set_entry_point("step_a")
    graph.add_edge("step_a", "step_b")
    graph.add_edge("step_b", "step_c")
    graph.set_finish_point("step_c")

    app = graph.compile(checkpointer=saver)
    thread = "session-pregel-01"
    config = {"configurable": {"thread_id": thread}}

    print(f"  [+] Executing pipeline under thread_id='{thread}'...")
    final = app.invoke({"log": ["Initialized"]}, config=config)
    print(f"      Execution finished. Log: {final['log']}")

    history = saver.list_history(thread)
    print(f"  [+] Stored Checkpoints count: {len(history)}")
    for cp in history:
        print(f"      - CP {cp.checkpoint_id} (Superstep {cp.superstep}) | State: {cp.state.get('log')}")

    # Rewind to second checkpoint (Step A)
    if len(history) >= 2:
        target_cp = history[1]
        print(f"\n  [Time Travel] Rewinding thread to Checkpoint {target_cp.checkpoint_id}...")
        saver.rewind(thread, target_cp.checkpoint_id)
        rewound_history = saver.list_history(thread)
        print(f"      Rewound history size: {len(rewound_history)}")
        print(f"      Current active state: {rewound_history[-1].state['log']}\n")


def demonstrate_hitl_interrupt() -> None:
    print("-" * 68)
    print("3. Human-in-the-Loop (HITL): Breakpoint Pausing & Human Authorization")
    print("-" * 68)

    saver = MemorySaver()
    graph = StateGraph()

    def prepare_deployment(state: AgentState) -> AgentState:
        print("    [prepare_deployment] Staging artifacts in sandbox...")
        return {"status": "staged", "target_env": "production", "artifact": "v2.4.0-release"}

    def deploy_to_prod(state: AgentState) -> AgentState:
        print(f"    [deploy_to_prod] Executing production rollout with approval token: {state.get('approval_token')}")
        return {"status": "deployed_live", "deployed_at": "2026-10-21T14:30:00Z"}

    graph.add_node("prepare", prepare_deployment)
    graph.add_node("deploy", deploy_to_prod)
    graph.set_entry_point("prepare")
    graph.add_edge("prepare", "deploy")
    graph.set_finish_point("deploy")

    # Configure breakpoint BEFORE the sensitive deploy node
    app = graph.compile(checkpointer=saver, interrupt_before=["deploy"])
    thread_id = "prod-release-42"
    config = {"configurable": {"thread_id": thread_id}}

    print(f"  [+] Triggering pipeline with breakpoint before 'deploy'...")
    try:
        app.invoke({"status": "initiated"}, config=config)
    except GraphInterrupt as e:
        print(f"  [INTERRUPT DETECTED] {e}")
        print(f"      State at pause: {e.state}")

    # Simulated Human Review
    print("\n  [Human Reviewer] Inspecting staged artifacts: Pass.")
    print("  [Human Reviewer] Granting authorization token 'SEC-AUTH-9921' and resuming...")

    # Resume graph execution with approval token injection
    resumed_state = app.invoke({"approval_token": "SEC-AUTH-9921"}, config=config)
    print(f"  [Final Result] Deployment Status: {resumed_state['status']}")
    print(f"                 Target Environment: {resumed_state['target_env']}\n")


def main() -> None:
    print_banner()
    display_roadmap()
    demonstrate_cyclic_reflection()
    demonstrate_persistence_and_time_travel()
    demonstrate_hitl_interrupt()
    print("=" * 76)
    print("   LANGGRAPH PRODUCTION ENGINE DEMO COMPLETED SUCCESSFULLY   ")
    print("=" * 76)


if __name__ == "__main__":
    main()
