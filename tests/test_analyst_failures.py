from __future__ import annotations

from types import SimpleNamespace as Ns

from stoa import analyst as an


def msg(stop: str, text: str | None, out: int = 400, category: str | None = None) -> Ns:
    content = [Ns(type="thinking", thinking="")] + ([Ns(type="text", text=text)] if text else [])
    return Ns(
        stop_reason=stop,
        stop_details=Ns(category=category) if category else None,
        content=content,
        usage=Ns(
            input_tokens=900,
            output_tokens=out,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


GOOD = (
    '{"sentiment": 0.4, "guidance": "none", "surprise": "none", "risk_flags": [], '
    '"materiality": 0.3, "confidence": 0.7}'
)


def test_reasons_say_why_an_item_could_not_be_read() -> None:
    assert an.parse_message(msg("end_turn", GOOD)) is not None
    assert (
        an.failure_reason(msg("max_tokens", None, 400))
        == "cut off at max_tokens (400 output tokens)"
    )
    assert an.failure_reason(msg("refusal", None, 3, "cyber")) == "refusal (cyber)"
    assert an.failure_reason(msg("end_turn", '{"oops": 1}', 20)).startswith("answer does not match")
    assert an.failure_reason(msg("end_turn", None, 0)).startswith("no text block")


def test_failure_summary_groups_reasons_without_item_ids() -> None:
    fails = [
        "a1: unparsable, cut off at max_tokens (400 output tokens)",
        "a2: unparsable, cut off at max_tokens (400 output tokens)",
        "a3: errored (overloaded_error)",
        "a4: unparsable, refusal (cyber)",
    ]
    assert an.summarize_failures(fails) == (
        "unparsable, cut off at max_tokens x 2, errored (overloaded_error) x 1, "
        "unparsable, refusal (cyber) x 1"
    )


def test_batch_failures_rereads_results_and_lists_only_bad_items() -> None:
    lines = [
        Ns(custom_id="k1", result=Ns(type="succeeded", message=msg("end_turn", GOOD, 60))),
        Ns(custom_id="k2", result=Ns(type="succeeded", message=msg("max_tokens", None, 400))),
        Ns(custom_id="k3", result=Ns(type="expired")),
    ]
    client = Ns(messages=Ns(batches=Ns(results=lambda bid: iter(lines))))
    assert an.batch_failures(client, ["b1"]) == [
        "k2: unparsable, cut off at max_tokens (400 output tokens)",
        "k3: expired",
    ]


def test_the_analyst_has_room_to_think_before_its_answer() -> None:
    from pathlib import Path

    from stoa.config import load_settings

    assert load_settings(Path("config/default.yaml")).analyst.max_tokens >= 4000
