"""
Production-grade LangGraph Stateful Agent & Cyclic Graph Engine.

Implements the Pregel superstep execution model, StateGraph builder, channel reducers,
conditional branching, persistence checkpointing (MemorySaver), Human-in-the-Loop (HITL)
breakpoints, time-travel rewind, and multi-agent coordination.
"""

from __future__ import annotations
import copy
import operator
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

# Canonical LangGraph Sentinels
START = "__start__"
END = "__end__"

AgentState = Dict[str, Any]
NodeAction = Callable[[AgentState], Union[AgentState, Dict[str, Any]]]
RouterFunc = Callable[[AgentState], str]
ReducerFunc = Callable[[Any, Any], Any]


class RecursionLimitExceeded(RuntimeError):
    """Raised when the graph exceeds the configured superstep recursion limit."""
    pass


class GraphInterrupt(Exception):
    """Raised when a graph execution hits a human-in-the-loop breakpoint."""
    def __init__(self, node: str, state: AgentState, message: str = "Execution paused for human intervention"):
        super().__init__(f"{message} at node '{node}'")
        self.node = node
        self.state = state
        self.message = message


class Checkpoint:
    """Represents an immutable snapshot of state at a given superstep."""

    def __init__(
        self,
        checkpoint_id: str,
        thread_id: str,
        superstep: int,
        state: AgentState,
        next_nodes: List[str],
        interrupted: bool = False,
    ):
        self.checkpoint_id = checkpoint_id
        self.thread_id = thread_id
        self.superstep = superstep
        self.state = copy.deepcopy(state)
        self.next_nodes = list(next_nodes)
        self.interrupted = interrupted

    def __repr__(self) -> str:
        return (
            f"Checkpoint(id={self.checkpoint_id!r}, step={self.superstep}, "
            f"next={self.next_nodes}, interrupted={self.interrupted})"
        )


class Checkpointer(ABC := object):
    """Abstract interface for LangGraph state persistence."""

    def put(self, thread_id: str, checkpoint: Checkpoint) -> None:
        raise NotImplementedError

    def get(self, thread_id: str) -> Optional[Checkpoint]:
        raise NotImplementedError

    def list_history(self, thread_id: str) -> List[Checkpoint]:
        raise NotImplementedError


class MemorySaver(Checkpointer):
    """In-memory implementation of LangGraph persistent checkpointer (Write-Ahead Logging)."""

    def __init__(self):
        # thread_id -> list of Checkpoints in chronological order
        self._storage: Dict[str, List[Checkpoint]] = {}

    def put(self, thread_id: str, checkpoint: Checkpoint) -> None:
        if thread_id not in self._storage:
            self._storage[thread_id] = []
        self._storage[thread_id].append(checkpoint)

    def get(self, thread_id: str) -> Optional[Checkpoint]:
        history = self._storage.get(thread_id, [])
        return history[-1] if history else None

    def list_history(self, thread_id: str) -> List[Checkpoint]:
        return list(self._storage.get(thread_id, []))

    def rewind(self, thread_id: str, checkpoint_id: str) -> Optional[Checkpoint]:
        """Time-travel: locate a historical checkpoint and truncate forward history."""
        history = self._storage.get(thread_id, [])
        for idx, cp in enumerate(history):
            if cp.checkpoint_id == checkpoint_id:
                # Fork / rewind by keeping up to idx
                reverted = copy.deepcopy(cp)
                self._storage[thread_id] = history[:idx + 1]
                return reverted
        return None


class StateSchema:
    """Defines channel keys and optional reduction functions."""

    def __init__(self, reducers: Optional[Dict[str, ReducerFunc]] = None):
        # Default reducer is overwrite (last write wins)
        self.reducers: Dict[str, ReducerFunc] = reducers or {}

    def apply_delta(self, base_state: AgentState, delta: Dict[str, Any]) -> AgentState:
        new_state = copy.deepcopy(base_state)
        for key, val in delta.items():
            if key in self.reducers and key in new_state:
                reducer = self.reducers[key]
                new_state[key] = reducer(new_state[key], val)
            else:
                new_state[key] = copy.deepcopy(val)
        return new_state


class Edge:
    """Represents a direct or conditional transition between graph nodes."""

    def __init__(
        self,
        source: str,
        target: Optional[str] = None,
        router: Optional[RouterFunc] = None,
        path_map: Optional[Dict[str, str]] = None,
    ):
        self.source = source
        self.target = target
        self.router = router
        self.path_map = path_map or {}

    def resolve(self, state: AgentState) -> str:
        if self.target is not None:
            return self.target
        if self.router is not None:
            key = self.router(state)
            if self.path_map:
                if key not in self.path_map:
                    raise KeyError(f"Router returned key '{key}' not in path_map {list(self.path_map.keys())}")
                return self.path_map[key]
            return key
        raise RuntimeError(f"Edge from '{self.source}' has neither target nor router configured.")


class StateGraph:
    """
    Core graph builder for cyclic state machines.
    Supports nodes, edges, conditional edges, channel reducers, and Pregel compilation.
    """

    def __init__(self, schema: Optional[Union[Dict[str, ReducerFunc], StateSchema]] = None):
        if isinstance(schema, StateSchema):
            self.schema = schema
        elif isinstance(schema, dict):
            self.schema = StateSchema(reducers=schema)
        else:
            self.schema = StateSchema()

        self.nodes: Dict[str, NodeAction] = {}
        self.edges: Dict[str, List[Edge]] = {}
        self.entry_point: Optional[str] = None

    def add_node(self, name: str, action: NodeAction) -> StateGraph:
        if name in [START, END]:
            raise ValueError(f"Cannot name user node with reserved sentinel '{name}'")
        self.nodes[name] = action
        return self

    def set_entry_point(self, node: str) -> StateGraph:
        self.entry_point = node
        self.add_edge(START, node)
        return self

    def set_finish_point(self, node: str) -> StateGraph:
        self.add_edge(node, END)
        return self

    def add_edge(self, source: str, target: str) -> StateGraph:
        if source != START and source not in self.nodes:
            raise KeyError(f"Source node '{source}' not registered in graph")
        if target != END and target not in self.nodes:
            raise KeyError(f"Target node '{target}' not registered in graph")

        if source not in self.edges:
            self.edges[source] = []
        self.edges[source].append(Edge(source=source, target=target))
        return self

    def add_conditional_edges(
        self,
        source: str,
        path: RouterFunc,
        path_map: Optional[Dict[str, str]] = None,
    ) -> StateGraph:
        if source != START and source not in self.nodes:
            raise KeyError(f"Source node '{source}' not registered in graph")
        if source not in self.edges:
            self.edges[source] = []
        self.edges[source].append(Edge(source=source, router=path, path_map=path_map))
        return self

    def compile(
        self,
        checkpointer: Optional[Checkpointer] = None,
        interrupt_before: Optional[List[str]] = None,
        interrupt_after: Optional[List[str]] = None,
    ) -> CompiledGraph:
        if START not in self.edges or not self.edges[START]:
            raise ValueError("StateGraph must have an entry point configured via set_entry_point() or add_edge(START, ...)")
        return CompiledGraph(
            nodes=self.nodes,
            edges=self.edges,
            schema=self.schema,
            checkpointer=checkpointer,
            interrupt_before=set(interrupt_before or []),
            interrupt_after=set(interrupt_after or []),
        )


class CompiledGraph:
    """Compiled runnable execution engine for LangGraph state machine."""

    def __init__(
        self,
        nodes: Dict[str, NodeAction],
        edges: Dict[str, List[Edge]],
        schema: StateSchema,
        checkpointer: Optional[Checkpointer] = None,
        interrupt_before: Optional[Set[str]] = None,
        interrupt_after: Optional[Set[str]] = None,
    ):
        self.nodes = nodes
        self.edges = edges
        self.schema = schema
        self.checkpointer = checkpointer
        self.interrupt_before = interrupt_before or set()
        self.interrupt_after = interrupt_after or set()

    def invoke(
        self,
        input_data: AgentState,
        config: Optional[Dict[str, Any]] = None,
        recursion_limit: int = 25,
    ) -> AgentState:
        """Run graph to completion or until an interrupt breakpoint."""
        last_state = input_data
        for event in self.stream(input_data, config=config, recursion_limit=recursion_limit):
            last_state = event["state"]
        return last_state

    def stream(
        self,
        input_data: AgentState,
        config: Optional[Dict[str, Any]] = None,
        recursion_limit: int = 25,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream execution events superstep by superstep."""
        config = config or {}
        thread_id = config.get("configurable", {}).get("thread_id")

        current_state: AgentState = {}
        next_nodes: List[str] = []
        superstep = 0

        is_resuming_from_interrupt = False
        # Check for persistent resumption
        if thread_id and self.checkpointer:
            latest = self.checkpointer.get(thread_id)
            if latest:
                current_state = latest.state
                next_nodes = list(latest.next_nodes)
                superstep = latest.superstep
                if latest.interrupted:
                    is_resuming_from_interrupt = True
                # If fresh inputs provided on resume, merge them in
                if input_data:
                    current_state = self.schema.apply_delta(current_state, input_data)

        if not current_state:
            current_state = copy.deepcopy(input_data)
            # Resolve entry point from START
            start_edges = self.edges.get(START, [])
            if not start_edges:
                raise RuntimeError("No START edge configured")
            first_target = start_edges[0].resolve(current_state)
            next_nodes = [first_target]

        while next_nodes:
            superstep += 1
            if superstep > recursion_limit:
                raise RecursionLimitExceeded(
                    f"Graph exceeded recursion_limit of {recursion_limit} supersteps. Possible infinite loop."
                )

            current_node = next_nodes.pop(0)
            if current_node == END:
                break

            # HITL: interrupt_before check
            if current_node in self.interrupt_before:
                if is_resuming_from_interrupt:
                    # Bypassed since we are explicitly resuming after human approval
                    is_resuming_from_interrupt = False
                else:
                    if thread_id and self.checkpointer:
                        cp = Checkpoint(
                            checkpoint_id=str(uuid.uuid4())[:8],
                            thread_id=thread_id,
                            superstep=superstep,
                            state=current_state,
                            next_nodes=[current_node],
                            interrupted=True,
                        )
                        self.checkpointer.put(thread_id, cp)
                    raise GraphInterrupt(node=current_node, state=current_state, message="Interrupted BEFORE node execution")

            action = self.nodes.get(current_node)
            if action is None:
                raise KeyError(f"Node '{current_node}' not registered in nodes registry")

            # Execute node action
            delta = action(current_state)
            if isinstance(delta, dict):
                current_state = self.schema.apply_delta(current_state, delta)

            # Checkpoint after state modification
            if thread_id and self.checkpointer:
                cp = Checkpoint(
                    checkpoint_id=str(uuid.uuid4())[:8],
                    thread_id=thread_id,
                    superstep=superstep,
                    state=current_state,
                    next_nodes=[],
                    interrupted=False,
                )
                self.checkpointer.put(thread_id, cp)

            # HITL: interrupt_after check
            if current_node in self.interrupt_after:
                # Find subsequent routes before pausing
                resolved_subsequents = self._resolve_next(current_node, current_state)
                if thread_id and self.checkpointer:
                    cp_int = Checkpoint(
                        checkpoint_id=str(uuid.uuid4())[:8],
                        thread_id=thread_id,
                        superstep=superstep,
                        state=current_state,
                        next_nodes=resolved_subsequents,
                        interrupted=True,
                    )
                    self.checkpointer.put(thread_id, cp_int)
                raise GraphInterrupt(node=current_node, state=current_state, message="Interrupted AFTER node execution")

            # Route to next nodes
            resolved_next = self._resolve_next(current_node, current_state)
            for nxt in resolved_next:
                if nxt != END:
                    next_nodes.append(nxt)

            yield {
                "superstep": superstep,
                "node": current_node,
                "state": copy.deepcopy(current_state),
                "next": list(next_nodes)
            }

        # Terminal checkpoint
        if thread_id and self.checkpointer:
            terminal_cp = Checkpoint(
                checkpoint_id=str(uuid.uuid4())[:8],
                thread_id=thread_id,
                superstep=superstep + 1,
                state=current_state,
                next_nodes=[END],
                interrupted=False,
            )
            self.checkpointer.put(thread_id, terminal_cp)

        yield {
            "superstep": superstep + 1,
            "node": END,
            "state": current_state,
            "next": []
        }

    def _resolve_next(self, current_node: str, state: AgentState) -> List[str]:
        routes = self.edges.get(current_node, [])
        next_targets: List[str] = []
        for edge in routes:
            target = edge.resolve(state)
            next_targets.append(target)
        return next_targets
