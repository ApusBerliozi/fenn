import warnings

import pytest

from fenn.agents import (
    BaseNode,
    Flow,
    Node,
)


class TestBaseNode:
    def test_init_sets_empty_params_and_successors(self):
        node = BaseNode()
        assert node.params == {}
        assert node.successors == {}

    def test_set_params(self):
        node = BaseNode()
        node.set_params({"a": 1})
        assert node.params == {"a": 1}

    def test_default_prep_exec_post_return_none(self):
        node = BaseNode()
        assert node.prep({}) is None
        assert node.exec(None) is None
        assert node.post({}, None, None) is None

    def test_run_calls_prep_exec_post_in_order(self):
        calls = []

        class Tracked(BaseNode):
            def prep(self, shared):
                calls.append("prep")
                return "prepped"

            def exec(self, prep_res):
                calls.append(("exec", prep_res))
                return "executed"

            def post(self, shared, prep_res, exec_res):
                calls.append(("post", prep_res, exec_res))
                return "done"

        node = Tracked()
        result = node.run({})
        assert calls == ["prep", ("exec", "prepped"), ("post", "prepped", "executed")]
        assert result == "done"

    def test_run_warns_when_successors_present(self):
        node = BaseNode()
        node.successors["default"] = BaseNode()
        with pytest.warns(UserWarning, match="won't run successors"):
            node.run({})

    def test_run_no_warning_without_successors(self):
        node = BaseNode()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            node.run({})  # should not raise


class TestNode:
    def test_default_max_retries_and_wait(self):
        node = Node()
        assert node.max_retries == 1
        assert node.wait == 0

    def test_exec_succeeds_first_try(self):
        class Succeeds(Node):
            def exec(self, prep_res):
                return "ok"

        node = Succeeds()
        assert node._exec(None) == "ok"

    def test_exec_fallback_raises_by_default(self):
        class AlwaysFails(Node):
            def exec(self, prep_res):
                raise ValueError("boom")

        node = AlwaysFails(max_retries=1)
        with pytest.raises(ValueError, match="boom"):
            node._exec(None)

    def test_retries_then_succeeds(self):
        attempts = {"count": 0}

        class FlakyThenWorks(Node):
            def exec(self, prep_res):
                attempts["count"] += 1
                if attempts["count"] < 3:
                    raise RuntimeError("transient")
                return "success"

        node = FlakyThenWorks(max_retries=5)
        result = node._exec(None)
        assert result == "success"
        assert attempts["count"] == 3

    def test_exec_fallback_called_after_exhausting_retries(self):
        class FailsWithFallback(Node):
            def exec(self, prep_res):
                raise ValueError("always fails")

            def exec_fallback(self, prep_res, exc):
                return f"fallback: {exc}"

        node = FailsWithFallback(max_retries=2)
        result = node._exec(None)
        assert result == "fallback: always fails"

    def test_wait_triggers_sleep_between_retries(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr("fenn.agents.time.sleep", lambda s: sleeps.append(s))

        class AlwaysFails(Node):
            def exec(self, prep_res):
                raise ValueError("fail")

            def exec_fallback(self, prep_res, exc):
                return "fallback"

        node = AlwaysFails(max_retries=3, wait=2)
        node._exec(None)
        assert sleeps == [2, 2]

    def test_cur_retry_tracks_attempt_number(self):
        attempts = []

        class TrackRetry(Node):
            def exec(self, prep_res):
                attempts.append(self.cur_retry)
                raise ValueError("fail")

            def exec_fallback(self, prep_res, exc):
                return "fallback"

        node = TrackRetry(max_retries=3)
        node._exec(None)
        assert attempts == [0, 1, 2]

    def test_full_run_pipeline(self):
        class Adder(Node):
            def prep(self, shared):
                return shared["value"]

            def exec(self, prep_res):
                return prep_res + 1

            def post(self, shared, prep_res, exec_res):
                shared["result"] = exec_res
                return "done"

        node = Adder()
        shared = {"value": 5}
        action = node._run(shared)
        assert shared["result"] == 6
        assert action == "done"


class _Echo(BaseNode):
    """Simple node: records its label and returns a fixed next action.

    Uses instance attributes (not self.params) because Flow._orch()
    overwrites node.params on every step.
    """

    def __init__(self, label="node", next_action="default"):
        super().__init__()
        self.label = label
        self.next_action = next_action

    def post(self, shared, prep_res, exec_res):
        shared.setdefault("visited", []).append(self.label)
        return self.next_action


class TestFlow:
    def test_start_sets_start_node_and_returns_it(self):
        flow = Flow()
        node = BaseNode()
        result = flow.start(node)
        assert flow.start_node is node
        assert result is node

    def test_connect_returns_self(self):
        flow = Flow()
        a, b = BaseNode(), BaseNode()
        result = flow.connect(a, b)
        assert result is flow

    def test_connect_sets_successor(self):
        flow = Flow()
        a, b = BaseNode(), BaseNode()
        flow.connect(a, b, action="next")
        assert a.successors["next"] is b

    def test_connect_default_action(self):
        flow = Flow()
        a, b = BaseNode(), BaseNode()
        flow.connect(a, b)
        assert a.successors["default"] is b

    def test_connect_none_dest_sets_terminal(self):
        flow = Flow()
        a = BaseNode()
        flow.connect(a, None, action="done")
        assert flow.get_next_node(a, "done") is None

    def test_connect_overwrite_warns(self):
        flow = Flow()
        a, b, c = BaseNode(), BaseNode(), BaseNode()
        flow.connect(a, b, action="x")
        with pytest.warns(UserWarning, match="Overwriting successor"):
            flow.connect(a, c, action="x")

    def test_get_next_node_returns_successor(self):
        flow = Flow()
        a, b = BaseNode(), BaseNode()
        flow.connect(a, b, action="go")
        assert flow.get_next_node(a, "go") is b

    def test_get_next_node_defaults_to_default_action(self):
        flow = Flow()
        a, b = BaseNode(), BaseNode()
        flow.connect(a, b)
        assert flow.get_next_node(a, None) is b

    def test_get_next_node_unknown_action_warns(self):
        flow = Flow()
        a, b = BaseNode(), BaseNode()
        flow.connect(a, b, action="default")
        with pytest.warns(UserWarning, match="Flow ends"):
            result = flow.get_next_node(a, "unknown")
        assert result is None

    def test_get_next_node_no_successors_no_warning(self):
        flow = Flow()
        a = BaseNode()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = flow.get_next_node(a, "anything")
        assert result is None

    def test_orch_traverses_chain(self):
        a = _Echo(label="a")
        b = _Echo(label="b")
        c = _Echo(label="c")
        flow = Flow(start=a)
        flow.connect(a, b).connect(b, c).connect(c, None)

        shared = {}
        flow._orch(shared)
        assert shared["visited"] == ["a", "b", "c"]

    def test_post_returns_exec_res(self):
        flow = Flow()
        assert flow.post({}, "prep", "exec_result") == "exec_result"

    def test_full_flow_run(self):
        a = _Echo(label="a", next_action="next")
        b = _Echo(label="b", next_action="default")

        flow = Flow(start=a)
        flow.connect(a, b, action="next")
        flow.connect(b, None)

        shared = {}
        result = flow._run(shared)
        assert shared["visited"] == ["a", "b"]
        assert result == "default"

    def test_branching_flow(self):
        """Flow takes different paths based on returned action."""

        class Router(BaseNode):
            def post(self, shared, prep_res, exec_res):
                shared.setdefault("visited", []).append("router")
                return shared["route"]

        class Leaf(BaseNode):
            def __init__(self, name):
                super().__init__()
                self.name = name

            def post(self, shared, prep_res, exec_res):
                shared.setdefault("visited", []).append(self.name)
                return "default"

        router = Router()
        left = Leaf("left")
        right = Leaf("right")

        flow = Flow(start=router)
        flow.connect(router, left, action="left")
        flow.connect(router, right, action="right")
        flow.connect(left, None)
        flow.connect(right, None)

        shared = {"route": "right"}
        flow._orch(shared)
        assert shared["visited"] == ["router", "right"]


class TestModuleExports:
    def test_llmclient_importable(self):
        from fenn.agents import LLMClient

        assert LLMClient is not None

    def test_ragnode_importable(self):
        from fenn.agents import RAGNode

        assert RAGNode is not None

    def test_all_exports_present(self):
        import fenn.agents as agents_module

        for name in agents_module.__all__:
            assert hasattr(agents_module, name), f"{name} missing from module"
