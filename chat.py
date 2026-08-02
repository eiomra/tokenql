"""Plain user/assistant interface for TokenQL's inference backends."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from streaming_qwen import StreamingModelError
from tokenql import (
    Session,
    TokenQLEngine,
    TokenQLError,
    add_backend_arguments,
    create_backend,
)

DEFAULT_SYSTEM_PROMPT = ""


def initial_history(system_prompt: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system_prompt}] if system_prompt.strip() else []


def generation_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "strategy": args.strategy,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        **({"seed": args.seed} if args.seed is not None else {}),
    }


def answer(
    engine: TokenQLEngine,
    prompt: str,
    history: list[dict[str, str]],
    options: dict[str, Any],
    on_update: Any = None,
) -> str:
    messages = [*history, {"role": "user", "content": prompt}]
    if on_update is None:
        return engine.generate_text(prompt, options, messages=messages)
    return engine.generate_text(prompt, options, messages=messages, on_update=on_update)


def answer_session(
    engine: TokenQLEngine,
    session: Session,
    prompt: str,
    history: list[dict[str, str]],
    options: dict[str, Any],
    on_update: Any = None,
) -> str:
    messages = [*history, {"role": "user", "content": prompt}]
    started = time.perf_counter()
    reads_before = getattr(getattr(engine.backend, "store", None), "bytes_read", 0)
    if session.initialized:
        prepare = getattr(engine.backend, "prepare_continuation", None)
        if prepare is not None:
            prepare(session, messages)
        else:
            release = getattr(engine.backend, "release", None)
            if release is not None:
                release(session)
            session.prompt = prompt
            session.messages = messages
            session.prompt_ids = []
            session.generated_ids = []
            session.finished = False
            session.next_logits = None
            session.initialized = False
    else:
        session.prompt = prompt
        session.messages = messages
    return engine.generate_session(
        session,
        options,
        on_update=on_update,
        started_at=started,
        reads_before=reads_before,
    )


def print_stats(engine: TokenQLEngine) -> None:
    result = getattr(engine, "last_generation", None)
    if not result:
        return
    tokens = int(result["new_tokens"])
    elapsed = float(result["elapsed_seconds"])
    rate = tokens / elapsed if elapsed else 0.0
    gib = int(result.get("logical_weight_bytes", 0)) / (1024**3)
    suffix = ""
    prefill_layout_line = ""
    prefill_seconds = float(result.get("prefill_seconds", 0.0))
    prefill_tokens = int(result.get("prefill_tokens", 0))
    decode_seconds = float(result.get("decode_seconds", 0.0))
    decode_steps = int(result.get("decode_steps", 0))
    if prefill_tokens:
        suffix += f", prefill {prefill_tokens} tokens/{prefill_seconds:.2f}s"
    if decode_steps and decode_seconds:
        suffix += f", decode {decode_steps / decode_seconds:.2f} tok/s"
    if "decode_cache_hit_rate" in result:
        suffix += f", decode cache {float(result['decode_cache_hit_rate']) * 100:.1f}% hit"
    if "decode_prefetch_wait_seconds" in result:
        suffix += f", decode I/O wait {float(result['decode_prefetch_wait_seconds']):.2f}s"
    if hasattr(engine.backend, "stats"):
        runtime = engine.backend.stats()
        shared = int(runtime.get("resident_shared_matrix_bytes", 0))
        if shared:
            suffix += f", shared cache {shared / 1048576:.1f} MiB"
        cache = runtime.get("expert_cache")
        if cache:
            suffix += (
                f", expert cache {cache['hit_rate'] * 100:.1f}% hit, "
                f"{cache['resident_bytes'] / 1048576:.1f}/"
                f"{cache['capacity_bytes'] / 1048576:.1f} MiB"
            )
            prefetch = cache.get("prefetch", {})
            if prefetch.get("submitted"):
                suffix += (
                    f", prefetch {prefetch['useful_rate'] * 100:.1f}% useful "
                    f"({prefetch['ready']} ready/{prefetch['waits']} waited, "
                    f"{float(prefetch.get('wait_seconds', 0.0)):.2f}s)"
                )
            prediction = cache.get("prediction", {})
            if prediction.get("candidates"):
                suffix += (
                    f", predictor {prediction['accuracy'] * 100:.1f}% accurate "
                    f"({prediction.get('correct', 0)}/"
                    f"{prediction.get('candidates', 0)} correct, "
                    f"{prediction.get('io_hits', 0)} I/O hits)"
                )
        stability = runtime.get("route_stability", {})
        measured = []
        for depth in ("1", "2", "3"):
            values = stability.get(depth, {})
            if values.get("observations"):
                measured.append(
                    f"d{depth} {float(values.get('precision', 0.0)) * 100:.1f}%/"
                    f"{float(values.get('candidates_per_layer', 0.0)):.1f} "
                    f"(top2 {float(values.get('top2_precision', 0.0)) * 100:.1f}%)"
                )
        if measured:
            suffix += ", route stability " + ", ".join(measured) + " precision/candidates"
        layout = runtime.get("prefill_layout", {})
        if layout.get("selected_experts"):
            policies = layout.get("policies", {})
            choices = []
            for gap in ("0", "1", "2", "4", "8"):
                values = policies.get(gap, {})
                choices.append(
                    f"g{gap} {float(values.get('read_fraction', 0.0)) * 100:.1f}% "
                    f"reads/{float(values.get('amplification', 0.0)):.2f}x bytes"
                )
            prefill_layout_line = (
                f"[prefill layout: {layout['selected_experts']} uncached experts, "
                f"{int(layout.get('exact_bytes', 0)) / 1048576:.1f} MiB exact; "
                + ", ".join(choices)
                + f", full-span {float(layout.get('full_span_amplification', 0.0)):.2f}x bytes]"
            )
    print(
        f"[stats: {tokens} tokens, {elapsed:.2f}s, {rate:.2f} tok/s, "
        f"{gib:.2f} GiB weight reads{suffix}]"
    )
    if prefill_layout_line:
        print(prefill_layout_line)


def thinking_enabled(engine: TokenQLEngine) -> bool:
    return bool(getattr(engine.backend, "thinking_enabled", False))


def set_thinking(engine: TokenQLEngine, enabled: bool) -> None:
    setter = getattr(engine.backend, "set_thinking", None)
    if setter is None:
        raise TokenQLError("This backend does not support changing thinking mode")
    setter(enabled)


def run_chat(
    engine: TokenQLEngine,
    system_prompt: str,
    options: dict[str, Any],
    show_stats: bool,
) -> int:
    history: list[dict[str, str]] = initial_history(system_prompt)
    conversation = Session("__chat__", "", engine.backend.model_name)
    print(
        f"TokenQL Chat | backend={engine.backend.backend_name} | model={engine.backend.model_name}"
    )
    state = "on" if thinking_enabled(engine) else "off"
    print("Commands: /new clears the conversation, /thinking on|off changes reasoning, /exit quits")
    print(f"Thinking: {state}")
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            if hasattr(engine.backend, "release"):
                engine.backend.release(conversation)
            return 0
        if not prompt:
            continue
        if prompt.lower() in {"/exit", "/quit", "exit", "quit"}:
            if hasattr(engine.backend, "release"):
                engine.backend.release(conversation)
            return 0
        if prompt.lower() == "/new":
            if hasattr(engine.backend, "release"):
                engine.backend.release(conversation)
            conversation = Session("__chat__", "", engine.backend.model_name)
            history = initial_history(system_prompt)
            print("Conversation cleared.")
            continue
        thinking_command = prompt.lower().split()
        if thinking_command and thinking_command[0] == "/thinking":
            if len(thinking_command) == 1:
                state = "on" if thinking_enabled(engine) else "off"
                print(f"Thinking is {state}.")
            elif len(thinking_command) == 2 and thinking_command[1] in {"on", "off"}:
                enabled = thinking_command[1] == "on"
                try:
                    set_thinking(engine, enabled)
                except TokenQLError as exc:
                    print(f"error: {exc}", file=sys.stderr)
                else:
                    print(f"Thinking turned {'on' if enabled else 'off'}.")
            else:
                print("Usage: /thinking on|off")
            continue
        emitted = ""

        def show_update(full_text: str) -> None:
            nonlocal emitted
            # Byte-fallback tokens can temporarily decode as U+FFFD until the
            # following token completes a UTF-8 character. Hold that unstable
            # suffix instead of printing a replacement character that cannot
            # be retracted from a terminal stream.
            replacement = full_text.find("\ufffd")
            stable_text = full_text if replacement < 0 else full_text[:replacement]
            if stable_text.startswith(emitted):
                print(stable_text[len(emitted) :], end="", flush=True)
            emitted = stable_text

        print("ai> ", end="", flush=True)
        try:
            reply = answer_session(
                engine,
                conversation,
                prompt,
                history,
                options,
                on_update=show_update,
            )
        except (TokenQLError, StreamingModelError) as exc:
            print()
            print(f"error: {exc}", file=sys.stderr)
            continue
        if reply.startswith(emitted):
            print(reply[len(emitted) :], end="", flush=True)
        history.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reply},
            ]
        )
        print()
        if show_stats:
            print_stats(engine)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Plain chat interface for TokenQL")
    add_backend_arguments(parser)
    parser.add_argument("-p", "--prompt", help="Ask once and print only the answer")
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--strategy", choices=["sample", "greedy"], default="sample")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--stats", action="store_true", help="Show token rate and logical weight I/O"
    )
    args = parser.parse_args(argv)

    try:
        backend = create_backend(args, parser)
        engine = TokenQLEngine(backend, repository=None)
        options = generation_options(args)
        if args.prompt is not None:
            history = initial_history(args.system)
            print(answer(engine, args.prompt, history, options))
            if args.stats:
                print_stats(engine)
            return 0
        return run_chat(engine, args.system, options, args.stats)
    except (TokenQLError, StreamingModelError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
