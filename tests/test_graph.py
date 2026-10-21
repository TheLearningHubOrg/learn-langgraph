"""
Unit test suite for the LangGraph Stateful Agent & Cyclic Engine.
Verifies cyclic loops, channel reducers, checkpoint persistence, time-travel, and HITL.
"""

import operator
import unittest
from src.learn_langgraph.graph import (
    START,
    END,
    StateGraph,
    MemorySaver,
    GraphInterrupt,
    RecursionLimitExceeded,
    AgentState
)
from src.learn_langgraph.roadmap import (
    LANGGRAPH_STAGES,
    get_completed_count,
    get_progress_percentage
)


class TestLangGraphEngine(unittest.TestCase):

    def test_linear_graph_execution(self):
        graph = StateGraph()
        graph.add_node("step1", lambda s: {"val": s["val"] + 10})
        graph.add_node("step2", lambda s: {"val": s["val"] * 2})
        graph.set_entry_point("step1")
        graph.add_edge("step1", "step2")
        graph.set_finish_point("step2")

        app = graph.compile()
        result = app.invoke({"val": 5})
        self.assertEqual(result["val"], 30)

    def test_channel_reducers_list_append(self):
        # Uses operator.add reducer for 'items' channel
        graph = StateGraph(schema={"items": operator.add})
        graph.add_node("a", lambda s: {"items": ["from_a"]})
        graph.add_node("b", lambda s: {"items": ["from_b"]})
        graph.set_entry_point("a")
        graph.add_edge("a", "b")
        graph.set_finish_point("b")

        app = graph.compile()
        result = app.invoke({"items": ["initial"]})
        self.assertEqual(result["items"], ["initial", "from_a", "from_b"])

    def test_cyclic_graph_loop_and_termination(self):
        graph = StateGraph()

        def counter_node(s: AgentState) -> AgentState:
            return {"count": s.get("count", 0) + 1}

        def route_condition(s: AgentState) -> str:
            if s["count"] >= 3:
                return END
            return "counter"

        graph.add_node("counter", counter_node)
        graph.set_entry_point("counter")
        graph.add_conditional_edges("counter", route_condition)

        app = graph.compile()
        result = app.invoke({"count": 0})
        self.assertEqual(result["count"], 3)

    def test_recursion_limit_protection(self):
        graph = StateGraph()
        graph.add_node("infinite", lambda s: {"tick": s.get("tick", 0) + 1})
        graph.set_entry_point("infinite")
        graph.add_edge("infinite", "infinite")

        app = graph.compile()
        with self.assertRaises(RecursionLimitExceeded):
            app.invoke({"tick": 0}, recursion_limit=5)

    def test_memory_saver_persistence(self):
        saver = MemorySaver()
        graph = StateGraph()
        graph.add_node("n1", lambda s: {"stage": "stage_1"})
        graph.add_node("n2", lambda s: {"stage": "stage_2"})
        graph.set_entry_point("n1")
        graph.add_edge("n1", "n2")
        graph.set_finish_point("n2")

        app = graph.compile(checkpointer=saver)
        thread = "test-thread-wal"
        res = app.invoke({"stage": "init"}, config={"configurable": {"thread_id": thread}})

        self.assertEqual(res["stage"], "stage_2")
        history = saver.list_history(thread)
        self.assertTrue(len(history) >= 2)

    def test_time_travel_rewind(self):
        saver = MemorySaver()
        graph = StateGraph()
        graph.add_node("s1", lambda s: {"path": s.get("path", []) + ["s1"]})
        graph.add_node("s2", lambda s: {"path": s.get("path", []) + ["s2"]})
        graph.set_entry_point("s1")
        graph.add_edge("s1", "s2")
        graph.set_finish_point("s2")

        app = graph.compile(checkpointer=saver)
        thread = "time-travel-thread"
        app.invoke({"path": ["start"]}, config={"configurable": {"thread_id": thread}})

        history = saver.list_history(thread)
        self.assertTrue(len(history) >= 2)
        target_cp = history[1]

        rewound = saver.rewind(thread, target_cp.checkpoint_id)
        self.assertIsNotNone(rewound)
        self.assertEqual(rewound.checkpoint_id, target_cp.checkpoint_id)

    def test_human_in_the_loop_interrupt_and_resume(self):
        saver = MemorySaver()
        graph = StateGraph()
        graph.add_node("node_a", lambda s: {"step": "a"})
        graph.add_node("critical_node", lambda s: {"step": "critical", "auth": s.get("auth")})
        graph.set_entry_point("node_a")
        graph.add_edge("node_a", "critical_node")
        graph.set_finish_point("critical_node")

        app = graph.compile(checkpointer=saver, interrupt_before=["critical_node"])
        thread = "hitl-thread-101"
        config = {"configurable": {"thread_id": thread}}

        # First run: should interrupt before critical_node
        with self.assertRaises(GraphInterrupt) as ctx:
            app.invoke({"auth": "none"}, config=config)
        self.assertEqual(ctx.exception.node, "critical_node")

        # Second run: resume with authorization payload
        resumed = app.invoke({"auth": "GRANTED_TOKEN_XYZ"}, config=config)
        self.assertEqual(resumed["step"], "critical")
        self.assertEqual(resumed["auth"], "GRANTED_TOKEN_XYZ")

    def test_roadmap_7_stages(self):
        self.assertEqual(len(LANGGRAPH_STAGES), 7)
        self.assertEqual(get_completed_count(), 7)
        self.assertEqual(get_progress_percentage(), 100)


if __name__ == "__main__":
    unittest.main()
