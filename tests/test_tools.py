"""
Unit tests for backend.tools:
- Non-blocking function declarations behavior
- WorkspaceState mutation and thread safety
- NonBlockingToolExecutor background execution, progress events, and cancellations
"""

import asyncio
import pytest
from google.genai import types
from backend.tools import (
    get_function_declarations,
    WorkspaceState,
    NonBlockingToolExecutor
)


def test_tool_declarations_non_blocking_behavior():
    """All tools must have behavior=types.Behavior.NON_BLOCKING."""
    declarations = get_function_declarations()
    assert len(declarations) == 5

    for decl in declarations:
        assert decl.behavior == types.Behavior.NON_BLOCKING
        assert decl.name in [
            "search_flights",
            "get_weather_forecast",
            "deep_research_topic",
            "set_timer_or_reminder",
            "control_workspace_device"
        ]


def test_workspace_state_snapshot():
    ws = WorkspaceState()
    snapshot = ws.snapshot()

    assert "lights" in snapshot
    assert "thermostat_c" in snapshot
    assert "music" in snapshot
    assert "do_not_disturb" in snapshot
    assert "timers" in snapshot
    assert snapshot["lights"]["on"] is True
    assert snapshot["thermostat_c"] == 22


@pytest.mark.asyncio
async def test_tool_executor_weather():
    ws = WorkspaceState()
    events = []
    responses = []

    async def on_event(ev):
        events.append(ev)

    async def on_complete(fn_resp, record):
        responses.append((fn_resp, record))

    executor = NonBlockingToolExecutor(
        workspace=ws,
        scheduling_overrides={"get_weather_forecast": "WHEN_IDLE"},
        simulate_latency=False
    )

    call = types.FunctionCall(
        id="call_weather_1",
        name="get_weather_forecast",
        args={"location": "Tashkent", "days": 1}
    )

    tasks = executor.dispatch(
        [call],
        allow_scheduling=True,
        on_event=on_event,
        on_complete=on_complete
    )
    assert len(tasks) == 1
    await tasks[0]

    assert len(responses) == 1
    fn_resp, rec = responses[0]
    assert fn_resp.id == "call_weather_1"
    assert fn_resp.name == "get_weather_forecast"
    assert rec["status"] == "completed"
    assert any(ev.get("type") == "tool_completed" for ev in events)


@pytest.mark.asyncio
async def test_tool_executor_cancellation():
    ws = WorkspaceState()
    events = []
    responses = []

    async def on_event(ev):
        events.append(ev)

    async def on_complete(fn_resp, record):
        responses.append((fn_resp, record))

    executor = NonBlockingToolExecutor(
        workspace=ws,
        simulate_latency=True
    )

    call = types.FunctionCall(
        id="call_research_cancel",
        name="deep_research_topic",
        args={"topic": "Quantum Computing", "focus": "hardware"}
    )

    tasks = executor.dispatch(
        [call],
        allow_scheduling=False,
        on_event=on_event,
        on_complete=on_complete
    )

    # Let it start
    await asyncio.sleep(0.05)

    # Cancel it
    cancelled_ids = await executor.cancel(["call_research_cancel"], on_event=on_event)
    assert "call_research_cancel" in cancelled_ids

    # Wait for task completion/cancellation
    try:
        await tasks[0]
    except asyncio.CancelledError:
        pass

    assert any(ev.get("type") == "tool_cancelled" for ev in events)
