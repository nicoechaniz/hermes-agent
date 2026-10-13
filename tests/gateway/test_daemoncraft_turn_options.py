"""Event tool choice reaches the current request route, never run_conversation kwargs."""
from types import SimpleNamespace

from gateway.platforms.event import MessageEvent
from gateway.turn_context import TurnContext
from gateway.run_turn_runner import TurnRunner


def test_message_event_and_turn_context_hold_event_scoped_tool_choice():
    event = MessageEvent(text="wake", tool_choice="required")
    assert event.tool_choice == "required"
    assert TurnContext(tool_choice=event.tool_choice).tool_choice == "required"


def test_event_tool_choice_layers_over_provider_request_options_for_one_turn():
    route = {"request_overrides": {"extra_body": {"verbosity": "low"}}}
    TurnRunner._apply_event_tool_choice(route, "required")

    agent = SimpleNamespace(
        request_overrides={"extra_body": {"verbosity": "low"}},
        _gateway_turn_request_overrides={},
    )
    TurnRunner._merge_turn_request_overrides(agent, route)

    assert agent.request_overrides == {
        "extra_body": {"verbosity": "low"},
        "tool_choice": "required",
    }

    next_route = {"request_overrides": {"extra_body": {"verbosity": "low"}}}
    TurnRunner._apply_event_tool_choice(next_route, None)
    TurnRunner._merge_turn_request_overrides(agent, next_route)
    assert agent.request_overrides == {"extra_body": {"verbosity": "low"}}
