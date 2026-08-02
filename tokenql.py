# SPDX-License-Identifier: MIT

"""TokenQL: a small transactional query layer over autoregressive LLM inference."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import warnings
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import torch

# Some global Python installations contain an optional chardet version that is
# newer than requests expects. It does not affect local model inference, so keep
# that third-party compatibility warning out of the TokenQL user interface.
warnings.filterwarnings(
    "ignore",
    message=r"urllib3 .* or chardet.*doesn't match a supported version!",
)

from transformers import AutoModelForCausalLM, AutoTokenizer

from streaming_qwen import StreamingBackend, StreamingModelError

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
QUOTED = r"'((?:''|[^'])*)'"


class TokenQLError(Exception):
    """An error that is safe to show to a TokenQL user."""


def _unquote(value: str) -> str:
    return value.replace("''", "'")


def _split_options(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    quoted = False
    index = 0
    while index < len(text):
        if text[index] == "'":
            if quoted and index + 1 < len(text) and text[index + 1] == "'":
                index += 2
                continue
            quoted = not quoted
        elif text[index] == "," and not quoted:
            parts.append(text[start:index].strip())
            start = index + 1
        index += 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def parse_options(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    options: dict[str, Any] = {}
    for part in _split_options(text):
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)", part)
        if not match:
            raise TokenQLError(f"Invalid option: {part}")
        key, raw = match.group(1).lower(), match.group(2).strip()
        if len(raw) >= 2 and raw[0] == raw[-1] == "'":
            value: Any = _unquote(raw[1:-1])
        elif raw.lower() in {"true", "false"}:
            value = raw.lower() == "true"
        else:
            try:
                value = float(raw) if any(char in raw.lower() for char in ".e") else int(raw)
            except ValueError as exc:
                raise TokenQLError(f"Invalid value for {key}: {raw}") from exc
        options[key] = value
    return options


def split_statements(source: str) -> list[str]:
    statements: list[str] = []
    start = 0
    quoted = False
    index = 0
    while index < len(source):
        if source[index] == "'":
            if quoted and index + 1 < len(source) and source[index + 1] == "'":
                index += 2
                continue
            quoted = not quoted
        elif source[index] == ";" and not quoted:
            statement = source[start:index].strip()
            if statement:
                statements.append(statement)
            start = index + 1
        index += 1
    remainder = source[start:].strip()
    if remainder:
        statements.append(remainder)
    return statements


@dataclass
class Session:
    session_id: str
    prompt: str
    model_name: str
    generated_ids: list[int] = field(default_factory=list)
    finished: bool = False
    prompt_ids: list[int] = field(default_factory=list, repr=False)
    cache: Any = field(default=None, repr=False)
    next_logits: torch.Tensor | None = field(default=None, repr=False)
    initialized: bool = field(default=False, repr=False)
    messages: list[dict[str, str]] | None = field(default=None, repr=False)
    prefill_seconds: float = field(default=0.0, repr=False)
    prefill_tokens: int = field(default=0, repr=False)
    decode_seconds: float = field(default=0.0, repr=False)
    decode_steps: int = field(default=0, repr=False)
    cache_hits_after_prefill: int = field(default=0, repr=False)
    cache_misses_after_prefill: int = field(default=0, repr=False)
    prefetch_wait_seconds_after_prefill: float = field(default=0.0, repr=False)


class SessionRepository:
    def __init__(self, path: str | Path):
        self.path = str(path)
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    generated_ids TEXT NOT NULL,
                    finished INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def insert(self, session: Session) -> None:
        try:
            with closing(self._connect()) as connection:
                connection.execute(
                    "INSERT INTO sessions(session_id, prompt, model_name, generated_ids, finished) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        session.session_id,
                        session.prompt,
                        session.model_name,
                        json.dumps(session.generated_ids),
                        int(session.finished),
                    ),
                )
                connection.commit()
        except sqlite3.IntegrityError as exc:
            raise TokenQLError(f"Session {session.session_id!r} already exists") from exc

    def save(self, session: Session) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                "UPDATE sessions SET generated_ids = ?, finished = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE session_id = ?",
                (json.dumps(session.generated_ids), int(session.finished), session.session_id),
            )
            connection.commit()

    def get(self, session_id: str) -> Session | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT session_id, prompt, model_name, generated_ids, finished "
                "FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return Session(
            session_id=row[0],
            prompt=row[1],
            model_name=row[2],
            generated_ids=json.loads(row[3]),
            finished=bool(row[4]),
        )

    def list(self) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT session_id, model_name, json_array_length(generated_ids), finished, updated_at "
                "FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [
            {
                "session_id": row[0],
                "model": row[1],
                "generated_tokens": row[2],
                "finished": bool(row[3]),
                "updated_at": row[4],
            }
            for row in rows
        ]

    def delete(self, session_id: str) -> bool:
        with closing(self._connect()) as connection:
            cursor = connection.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            connection.commit()
            return cursor.rowcount > 0


class TransformersBackend:
    """Direct model access, including logits and per-session DynamicCache objects."""

    backend_name = "transformers"

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        local_files_only: bool = False,
        thinking: bool = False,
    ):
        self.model_name = model_name
        self.device = torch.device(device)
        self.local_files_only = local_files_only
        self.thinking_enabled = bool(thinking)
        self._tokenizer: Any = None
        self._model: Any = None

    def set_thinking(self, enabled: bool) -> None:
        self.thinking_enabled = bool(enabled)

    @property
    def tokenizer(self):
        self._load()
        return self._tokenizer

    @property
    def model(self):
        self._load()
        return self._model

    def _load(self) -> None:
        if self._model is not None:
            return
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise TokenQLError(
                "CUDA was requested, but this PyTorch installation has no CUDA support"
            )
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, local_files_only=self.local_files_only
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
                dtype=dtype,
            ).to(self.device)
            self._model.eval()
        except Exception as exc:
            raise TokenQLError(f"Could not load model {self.model_name!r}: {exc}") from exc

    def _format_messages(self, messages: list[dict[str, str]]) -> list[int]:
        try:
            ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=self.thinking_enabled,
            )
        except (ValueError, TypeError):
            plain_text = "\n".join(
                f"{message['role']}: {message['content']}" for message in messages
            )
            ids = self.tokenizer(plain_text, return_tensors="pt").input_ids
        if ids.ndim == 2:
            ids = ids[0]
        return ids.tolist()

    def _format_prompt(self, prompt: str) -> list[int]:
        return self._format_messages(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )

    def initialize(self, session: Session, rebuild: bool = False) -> None:
        if session.initialized and not rebuild:
            return
        session.prompt_ids = (
            self._format_messages(session.messages)
            if session.messages is not None
            else self._format_prompt(session.prompt)
        )
        all_ids = session.prompt_ids + session.generated_ids
        input_ids = torch.tensor([all_ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.model(input_ids=input_ids, use_cache=True)
        session.cache = output.past_key_values
        session.next_logits = output.logits[0, -1].detach()
        session.initialized = True

    def eos_ids(self) -> set[int]:
        value = self.model.generation_config.eos_token_id
        if value is None:
            value = self.tokenizer.eos_token_id
        return set(value if isinstance(value, list) else [value])

    def distribution(
        self,
        session: Session,
        *,
        strategy: str = "sample",
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        self.initialize(session)
        if strategy not in {"sample", "greedy"}:
            raise TokenQLError("strategy must be 'sample' or 'greedy'")
        if temperature <= 0:
            raise TokenQLError("temperature must be greater than zero")
        if not 0 < top_p <= 1:
            raise TokenQLError("top_p must be in the interval (0, 1]")
        if top_k < 0:
            raise TokenQLError("top_k cannot be negative")

        logits = session.next_logits.float().clone()
        if strategy == "sample":
            logits /= temperature
            if top_k:
                keep = min(top_k, logits.numel())
                cutoff = torch.topk(logits, keep).values[-1]
                logits[logits < cutoff] = -torch.inf
            if top_p < 1:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = cumulative > top_p
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                logits[sorted_indices[remove]] = -torch.inf
        return torch.softmax(logits, dim=-1)

    def candidates(self, session: Session, count: int, **options: Any) -> list[dict[str, Any]]:
        if count < 1:
            raise TokenQLError("TOP must be at least 1")
        probabilities = self.distribution(session, **options)
        values, indices = torch.topk(probabilities, min(count, probabilities.numel()))
        return [
            self.describe_token(int(index), float(value))
            for value, index in zip(values, indices, strict=False)
        ]

    def choose(
        self, session: Session, *, seed: int | None = None, **options: Any
    ) -> dict[str, Any]:
        strategy = str(options.get("strategy", "sample"))
        probabilities = self.distribution(session, **options)
        if strategy == "greedy":
            token_id = int(torch.argmax(probabilities))
        else:
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            token_id = int(torch.multinomial(probabilities, 1, generator=generator))
        return self.describe_token(token_id, float(probabilities[token_id]))

    def describe_token(self, token_id: int, probability: float | None = None) -> dict[str, Any]:
        if not 0 <= token_id < len(self.tokenizer):
            raise TokenQLError(f"Token id {token_id} is outside this tokenizer's vocabulary")
        result: dict[str, Any] = {
            "token_id": token_id,
            "token": self.tokenizer.decode([token_id]),
            "token_piece": self.tokenizer.convert_ids_to_tokens(token_id),
        }
        if probability is not None:
            result["probability"] = probability
        return result

    def commit(self, session: Session, token_id: int, *, advance: bool = True) -> dict[str, Any]:
        self.initialize(session)
        if session.finished:
            raise TokenQLError("Session is finished; rewind it before appending more tokens")
        self.describe_token(token_id)
        finished = token_id in self.eos_ids()
        if advance and not finished:
            token = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
            with torch.inference_mode():
                output = self.model(input_ids=token, past_key_values=session.cache, use_cache=True)
            session.cache = output.past_key_values
            session.next_logits = output.logits[0, -1].detach()
        session.generated_ids.append(token_id)
        session.finished = finished
        result = self.describe_token(token_id)
        result.update(
            {
                "position": len(session.generated_ids) - 1,
                "committed": True,
                "finish_reason": "stop" if session.finished else None,
            }
        )
        return result

    def text(self, session: Session) -> str:
        self._load()
        return self.tokenizer.decode(session.generated_ids, skip_special_tokens=True)


class TokenQLEngine:
    ALLOWED_DECODING_OPTIONS: ClassVar[set[str]] = {
        "strategy",
        "temperature",
        "top_p",
        "top_k",
    }

    def __init__(self, backend: Any, repository: SessionRepository):
        self.backend = backend
        self.repository = repository
        self.sessions: dict[str, Session] = {}

    def _session(self, session_id: str) -> Session:
        if session_id not in self.sessions:
            session = self.repository.get(session_id)
            if session is None:
                raise TokenQLError(f"Unknown session {session_id!r}")
            if session.model_name != self.backend.model_name:
                raise TokenQLError(
                    f"Session uses {session.model_name!r}, but this process loaded {self.backend.model_name!r}"
                )
            self.sessions[session_id] = session
        return self.sessions[session_id]

    @staticmethod
    def _decoding_options(options: dict[str, Any]) -> dict[str, Any]:
        unknown = (
            set(options) - TokenQLEngine.ALLOWED_DECODING_OPTIONS - {"seed", "max_tokens", "stop"}
        )
        if unknown:
            raise TokenQLError(f"Unknown option(s): {', '.join(sorted(unknown))}")
        result = {
            key: options[key] for key in TokenQLEngine.ALLOWED_DECODING_OPTIONS if key in options
        }
        if "top_k" in result:
            result["top_k"] = int(result["top_k"])
        return result

    def _generate(
        self,
        session: Session,
        options: dict[str, Any],
        on_update: Any = None,
    ) -> dict[str, Any]:
        max_tokens = int(options.get("max_tokens", 128))
        if not 1 <= max_tokens <= 4096:
            raise TokenQLError("max_tokens must be between 1 and 4096")
        stop = str(options["stop"]) if "stop" in options else None
        decoding = self._decoding_options(options)
        seed = int(options["seed"]) if "seed" in options else None
        start = len(session.generated_ids)
        stopped_by_text = False
        for offset in range(max_tokens):
            chosen = self.backend.choose(
                session,
                seed=None if seed is None else seed + offset,
                **decoding,
            )
            self.backend.commit(
                session,
                chosen["token_id"],
                advance=offset + 1 < max_tokens,
            )
            current_text = self.backend.text(session)
            if on_update is not None:
                on_update(current_text)
            stopped_by_text = bool(stop and stop in current_text)
            if session.finished or stopped_by_text:
                break
        result = {
            "text": self.backend.text(session),
            "new_tokens": len(session.generated_ids) - start,
            "total_generated_tokens": len(session.generated_ids),
            "finish_reason": "stop" if session.finished or stopped_by_text else "length",
            "prefill_seconds": float(getattr(session, "prefill_seconds", 0.0)),
            "prefill_tokens": int(getattr(session, "prefill_tokens", 0)),
            "decode_seconds": float(getattr(session, "decode_seconds", 0.0)),
            "decode_steps": int(getattr(session, "decode_steps", 0)),
        }
        if hasattr(self.backend, "stats"):
            runtime = self.backend.stats()
            cache = runtime.get("expert_cache")
            if cache:
                decode_hits = max(0, int(cache["hits"]) - int(session.cache_hits_after_prefill))
                decode_misses = max(
                    0, int(cache["misses"]) - int(session.cache_misses_after_prefill)
                )
                attempts = decode_hits + decode_misses
                result["decode_cache_hit_rate"] = decode_hits / attempts if attempts else 0.0
                result["decode_prefetch_wait_seconds"] = max(
                    0.0,
                    float(cache.get("prefetch", {}).get("wait_seconds", 0.0))
                    - float(session.prefetch_wait_seconds_after_prefill),
                )
        return result

    def generate_session(
        self,
        session: Session,
        options: dict[str, Any],
        on_update: Any = None,
        *,
        started_at: float | None = None,
        reads_before: int | None = None,
    ) -> str:
        """Generate while retaining session/KV state for a later chat turn."""
        started = time.perf_counter() if started_at is None else float(started_at)
        if reads_before is None:
            reads_before = getattr(getattr(self.backend, "store", None), "bytes_read", 0)
        result = self._generate(session, options, on_update=on_update)
        result["elapsed_seconds"] = time.perf_counter() - started
        reads_after = getattr(getattr(self.backend, "store", None), "bytes_read", reads_before)
        result["logical_weight_bytes"] = reads_after - reads_before
        self.last_generation = result
        return result["text"]

    def generate_text(
        self,
        prompt: str,
        options: dict[str, Any],
        messages: list[dict[str, str]] | None = None,
        on_update: Any = None,
    ) -> str:
        session = Session(
            "__chat__",
            prompt,
            self.backend.model_name,
            messages=messages,
        )
        try:
            return self.generate_session(session, options, on_update=on_update)
        finally:
            if hasattr(self.backend, "release"):
                self.backend.release(session)

    def execute(self, statement: str) -> Any:
        sql = statement.strip().rstrip(";").strip()
        if not sql:
            return None

        if re.fullmatch(r"HELP", sql, re.IGNORECASE):
            return {"commands": HELP_COMMANDS}
        if re.fullmatch(r"SHOW\s+SESSIONS", sql, re.IGNORECASE):
            return self.repository.list()
        if re.fullmatch(r"SHOW\s+RUNTIME", sql, re.IGNORECASE):
            if hasattr(self.backend, "stats"):
                return self.backend.stats()
            return {"backend": "huggingface_transformers", "model": self.backend.model_name}

        create = re.fullmatch(
            rf"CREATE\s+(?:OR\s+REPLACE\s+)?SESSION\s+{QUOTED}(?:\s+USING\s+MODEL\s+{QUOTED})?\s+WITH\s+PROMPT\s+{QUOTED}",
            sql,
            re.IGNORECASE | re.DOTALL,
        )
        if create:
            replace = bool(re.match(r"CREATE\s+OR\s+REPLACE\s+SESSION", sql, re.IGNORECASE))
            session_id = _unquote(create.group(1))
            requested_model = (
                _unquote(create.group(2)) if create.group(2) else self.backend.model_name
            )
            prompt = _unquote(create.group(3))
            if requested_model != self.backend.model_name:
                raise TokenQLError(
                    f"This process serves {self.backend.model_name!r}; restart with --model {requested_model!r}"
                )
            if replace:
                previous = self.sessions.pop(session_id, None)
                if previous is not None and hasattr(self.backend, "release"):
                    self.backend.release(previous)
                self.repository.delete(session_id)
            session = Session(session_id, prompt, requested_model)
            self.repository.insert(session)
            self.sessions[session_id] = session
            return {"session_id": session_id, "model": requested_model, "created": True}

        one_shot = re.fullmatch(
            rf"SELECT\s+RESPONSE\s+FROM\s+(?:MODEL|QWEN)\s+WHERE\s+PROMPT\s*=\s*{QUOTED}(?:\s+WITH\s*\((.*)\))?",
            sql,
            re.IGNORECASE | re.DOTALL,
        )
        if one_shot:
            prompt = _unquote(one_shot.group(1))
            options = {
                "strategy": "sample",
                "temperature": 0.7,
                "top_p": 0.9,
                **parse_options(one_shot.group(2)),
            }
            return self.generate_text(prompt, options)

        drop = re.fullmatch(rf"DROP\s+SESSION\s+{QUOTED}", sql, re.IGNORECASE)
        if drop:
            session_id = _unquote(drop.group(1))
            previous = self.sessions.pop(session_id, None)
            if previous is not None and hasattr(self.backend, "release"):
                self.backend.release(previous)
            deleted = self.repository.delete(session_id)
            if not deleted:
                raise TokenQLError(f"Unknown session {session_id!r}")
            return {"session_id": session_id, "dropped": True}

        top = re.fullmatch(
            rf"SELECT\s+TOP\s+(\d+)\s+TOKEN\s*,\s*PROBABILITY\s+FROM\s+PREDICT_NEXT_TOKEN\s*\(\s*{QUOTED}\s*\)(?:\s+WITH\s*\((.*)\))?",
            sql,
            re.IGNORECASE | re.DOTALL,
        )
        if top:
            options = parse_options(top.group(3))
            return self.backend.candidates(
                self._session(_unquote(top.group(2))),
                int(top.group(1)),
                **self._decoding_options(options),
            )

        predict = re.fullmatch(
            rf"SELECT\s+NEXT_TOKEN\s+FROM\s+PREDICT_NEXT_TOKEN\s*\(\s*{QUOTED}\s*\)(?:\s+WITH\s*\((.*)\))?",
            sql,
            re.IGNORECASE | re.DOTALL,
        )
        if predict:
            options = parse_options(predict.group(2))
            chosen = self.backend.choose(
                self._session(_unquote(predict.group(1))),
                seed=int(options["seed"]) if "seed" in options else None,
                **self._decoding_options(options),
            )
            chosen.update({"committed": False, "finish_reason": None})
            return chosen

        append_token = re.fullmatch(
            rf"UPDATE\s+SESSION\s+{QUOTED}\s+APPEND\s+TOKEN\s+(\d+)",
            sql,
            re.IGNORECASE,
        )
        if append_token:
            session = self._session(_unquote(append_token.group(1)))
            result = self.backend.commit(session, int(append_token.group(2)))
            self.repository.save(session)
            return result

        append_next = re.fullmatch(
            rf"UPDATE\s+SESSION\s+{QUOTED}\s+APPEND\s+NEXT_TOKEN(?:\s+WITH\s*\((.*)\))?",
            sql,
            re.IGNORECASE | re.DOTALL,
        )
        if append_next:
            session = self._session(_unquote(append_next.group(1)))
            options = parse_options(append_next.group(2))
            chosen = self.backend.choose(
                session,
                seed=int(options["seed"]) if "seed" in options else None,
                **self._decoding_options(options),
            )
            result = self.backend.commit(session, chosen["token_id"])
            result["probability"] = chosen["probability"]
            self.repository.save(session)
            return result

        generate = re.fullmatch(
            rf"UPDATE\s+SESSION\s+{QUOTED}\s+GENERATE(?:\s+WITH\s*\((.*)\))?",
            sql,
            re.IGNORECASE | re.DOTALL,
        )
        if generate:
            session = self._session(_unquote(generate.group(1)))
            options = parse_options(generate.group(2))
            result = self._generate(session, options)
            self.repository.save(session)
            return result

        select_text = re.fullmatch(
            rf"SELECT\s+TEXT\s+FROM\s+SESSION\s*\(\s*{QUOTED}\s*\)",
            sql,
            re.IGNORECASE,
        )
        if select_text:
            session = self._session(_unquote(select_text.group(1)))
            return {
                "session_id": session.session_id,
                "text": self.backend.text(session),
                "generated_tokens": len(session.generated_ids),
                "finished": session.finished,
            }

        delete = re.fullmatch(
            rf"DELETE\s+TOKENS\s+FROM\s+SESSION\s+{QUOTED}\s+WHERE\s+POSITION\s*>=\s*(\d+)",
            sql,
            re.IGNORECASE,
        )
        if delete:
            session = self._session(_unquote(delete.group(1)))
            position = int(delete.group(2))
            removed = max(0, len(session.generated_ids) - position)
            session.generated_ids = session.generated_ids[:position]
            session.finished = False
            self.backend.initialize(session, rebuild=True)
            self.repository.save(session)
            return {
                "session_id": session.session_id,
                "removed_tokens": removed,
                "next_position": position,
            }

        raise TokenQLError("Unrecognized TokenQL statement. Enter HELP for supported syntax.")


HELP_COMMANDS = [
    "SELECT RESPONSE FROM MODEL WHERE PROMPT = 'your question';",
    "CREATE SESSION 'id' [USING MODEL 'name'] WITH PROMPT 'text';",
    "CREATE OR REPLACE SESSION 'id' WITH PROMPT 'text';",
    "DROP SESSION 'id';",
    "SELECT TOP 5 TOKEN, PROBABILITY FROM PREDICT_NEXT_TOKEN('id') WITH (temperature=0.8, top_p=0.95);",
    "SELECT NEXT_TOKEN FROM PREDICT_NEXT_TOKEN('id') WITH (strategy='greedy');",
    "UPDATE SESSION 'id' APPEND TOKEN 123;",
    "UPDATE SESSION 'id' APPEND NEXT_TOKEN WITH (strategy='greedy');",
    "UPDATE SESSION 'id' GENERATE WITH (max_tokens=128, temperature=0.8, top_p=0.95);",
    "SELECT TEXT FROM SESSION('id');",
    "DELETE TOKENS FROM SESSION 'id' WHERE POSITION >= 10;",
    "SHOW SESSIONS;",
    "SHOW RUNTIME;",
]


def _print_result(result: Any) -> None:
    if result is None:
        return
    if isinstance(result, str):
        print(result)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


def run_repl(engine: TokenQLEngine) -> int:
    print(
        f"TokenQL | backend={engine.backend.backend_name} | "
        f"model={engine.backend.model_name} | type HELP; to list commands"
    )
    buffer = ""
    while True:
        try:
            line = input("tokenql> " if not buffer else "      -> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not buffer and line.strip().lower() in {"quit", "exit", ".quit", ".exit"}:
            return 0
        buffer += ("\n" if buffer else "") + line
        if not line.rstrip().endswith(";"):
            continue
        try:
            for statement in split_statements(buffer):
                _print_result(engine.execute(statement))
        except (TokenQLError, StreamingModelError) as exc:
            print(f"error: {exc}", file=sys.stderr)
        buffer = ""


def add_backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["auto", "transformers", "streamed"], default="auto")
    parser.add_argument("--model", default=os.environ.get("TOKENQL_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ.get("TOKENQL_MODEL_DIR", "models/qwen2.5-0.5b-tokenql-q8")),
        help="Converted TokenQL model directory",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--weight-buffer-mb", type=int, default=64)
    parser.add_argument(
        "--ram-budget-mb",
        type=int,
        help=(
            "Hard budget for TokenQL-managed weight workspace, resident vectors, "
            "and compressed MoE experts (defaults to weight buffer only)"
        ),
    )
    parser.add_argument("--matmul", choices=["auto", "int8", "float"], default="auto")
    parser.add_argument(
        "--no-prefetch", action="store_true", help="Disable bounded asynchronous weight prefetch"
    )
    parser.add_argument(
        "--io-workers",
        type=int,
        default=1,
        help="Concurrent expert I/O workers for the streamed backend (default: 1)",
    )
    parser.add_argument(
        "--expert-prediction",
        action="store_true",
        help=(
            "Experimentally prefetch up to two routes stable across the prior "
            "two tokens (off by default)"
        ),
    )
    parser.add_argument(
        "--prefill-layout-profile",
        action="store_true",
        help="Report hypothetical MoE expert-read coalescing cost during prefill",
    )
    parser.add_argument(
        "--prefill-coalescing",
        action="store_true",
        help="Experimentally merge byte-adjacent prefill expert reads (off by default)",
    )
    parser.add_argument(
        "--prefill-coalescing-gap",
        type=int,
        choices=[0, 1, 2, 4, 8],
        default=0,
        help="Maximum skipped experts inside an experimental coalesced read",
    )
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--kv-cache-dir", type=Path)
    parser.add_argument("--threads", type=int, help="CPU tensor-kernel thread count")
    parser.add_argument(
        "--offline", action="store_true", help="Use only models already in the HF cache"
    )
    parser.add_argument(
        "--thinking",
        choices=["on", "off"],
        default="off",
        help="Enable or disable Qwen3 reasoning tokens (default: off)",
    )


def create_backend(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Any:
    if args.threads is not None:
        if args.threads < 1:
            parser.error("--threads must be positive")
        torch.set_num_threads(args.threads)
    if args.weight_buffer_mb < 1:
        parser.error("--weight-buffer-mb must be positive")
    if args.ram_budget_mb is not None and args.ram_budget_mb < args.weight_buffer_mb:
        parser.error("--ram-budget-mb cannot be smaller than --weight-buffer-mb")
    if not 1 <= args.io_workers <= 16:
        parser.error("--io-workers must be between 1 and 16")
    selected_backend = args.backend
    if selected_backend == "auto":
        selected_backend = (
            "streamed" if (args.model_dir / "tokenql_manifest.json").exists() else "transformers"
        )
    if selected_backend == "streamed":
        return StreamingBackend(
            args.model_dir,
            weight_buffer_mb=args.weight_buffer_mb,
            ram_budget_mb=args.ram_budget_mb,
            max_context=args.max_context,
            kv_cache_dir=args.kv_cache_dir,
            matmul=args.matmul,
            prefetch=not args.no_prefetch,
            thinking=args.thinking == "on",
            expert_prediction=args.expert_prediction,
            prefill_layout_profile=args.prefill_layout_profile,
            prefill_coalescing=args.prefill_coalescing,
            prefill_coalescing_gap=args.prefill_coalescing_gap,
            io_workers=args.io_workers,
        )
    return TransformersBackend(
        args.model,
        args.device,
        args.offline,
        thinking=args.thinking == "on",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transactional, SQL-like next-token inference")
    add_backend_arguments(parser)
    parser.add_argument("--db", default="tokenql.db", help="SQLite session-history database")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-e", "--execute", help="Execute one or more TokenQL statements")
    group.add_argument("-f", "--file", type=Path, help="Execute a TokenQL script")
    args = parser.parse_args(argv)

    try:
        backend = create_backend(args, parser)
        engine = TokenQLEngine(backend, SessionRepository(args.db))
        if args.execute is not None:
            for statement in split_statements(args.execute):
                _print_result(engine.execute(statement))
            return 0
        if args.file is not None:
            for statement in split_statements(args.file.read_text(encoding="utf-8")):
                _print_result(engine.execute(statement))
            return 0
        return run_repl(engine)
    except (TokenQLError, StreamingModelError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
