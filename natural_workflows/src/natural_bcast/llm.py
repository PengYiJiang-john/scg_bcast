from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sqlite3
import subprocess
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class Request:
    model: str
    temperature: float
    max_output_tokens: int
    purpose: str
    seed: int | None = None
    timeout_seconds: int = 240


class CachedLLM:
    def __init__(self, cache_path: Path, *, retries: int = 4, schema_retries: int = 2) -> None:
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.retries = retries
        self.schema_retries = schema_retries
        self._write_lock = threading.Lock()
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS calls (
                    cache_key TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    purpose TEXT NOT NULL,
                    model TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT,
                    usage_json TEXT,
                    status TEXT NOT NULL,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    model TEXT NOT NULL,
                    text_sha256 TEXT NOT NULL,
                    vector_json TEXT,
                    status TEXT NOT NULL,
                    error TEXT
                );
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.cache_path, timeout=60)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _digest(payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def generate_json(
        self,
        system: str,
        user: str,
        *,
        request: Request,
        schema: Mapping[str, Any],
    ) -> dict[str, Any]:
        cache_payload: dict[str, Any] = {
            "system": system,
            "user": user,
            "request": asdict(request),
            "schema": schema,
        }
        if "gemini-2.5-pro" in request.model.lower():
            cache_payload["generation_policy"] = (
                "pro-thinking-1024-with-separate-visible-output-budget-v1"
            )
        cache_key = self._digest(cache_payload)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT response_json FROM calls WHERE cache_key=? AND status='ok'",
                (cache_key,),
            ).fetchone()
        if row:
            return json.loads(row[0])

        generation: dict[str, Any] = {
            "temperature": request.temperature,
            "maxOutputTokens": request.max_output_tokens,
            "responseMimeType": "application/json",
            "responseSchema": dict(schema),
        }
        if request.seed is not None:
            generation["seed"] = request.seed
        if "flash" in request.model.lower():
            generation["thinkingConfig"] = {"thinkingBudget": 0}
        elif "gemini-2.5-pro" in request.model.lower():
            generation["thinkingConfig"] = {"thinkingBudget": 1024}
            generation["maxOutputTokens"] = request.max_output_tokens + 1024
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation,
        }
        response: dict[str, Any] | None = None
        parsed: dict[str, Any] | None = None
        last_error = ""
        attempts = self.retries + self.schema_retries + 1
        for attempt in range(attempts):
            if request.seed is not None:
                generation["seed"] = (request.seed + attempt) % (2**31 - 1)
            try:
                response = self._generate(request.model, payload, timeout=request.timeout_seconds)
                text = self._extract_text(response)
                parsed_value = self._parse_json_object(text)
                if not isinstance(parsed_value, dict):
                    raise LLMError("response is not an object")
                parsed = parsed_value
                break
            except (LLMError, json.JSONDecodeError, urllib.error.URLError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt + 1 >= attempts:
                    break
                delay = min(30.0, 1.25 * (2 ** min(attempt, 4)))
                delay += random.Random(f"{cache_key}:{attempt}").random()
                time.sleep(delay)

        usage = response.get("usageMetadata", {}) if response else {}
        record = {
            "cache_key": cache_key,
            "created_at": time.time(),
            "purpose": request.purpose,
            "model": request.model,
            "request_json": json.dumps(
                {
                    "system": system,
                    "user": user,
                    "request": asdict(request),
                    "schema": schema,
                },
                ensure_ascii=False,
            ),
            "response_json": json.dumps(parsed, ensure_ascii=False) if parsed is not None else None,
            "usage_json": json.dumps(usage, ensure_ascii=False),
            "status": "ok" if parsed is not None else "failed",
            "error": None if parsed is not None else last_error,
        }
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO calls
                (cache_key, created_at, purpose, model, request_json, response_json,
                 usage_json, status, error)
                VALUES (:cache_key, :created_at, :purpose, :model, :request_json,
                        :response_json, :usage_json, :status, :error)
                """,
                record,
            )
        if parsed is None:
            raise LLMError(f"request failed after {attempts} attempts: {last_error}")
        return parsed

    def embed(self, texts: Sequence[str], *, model: str) -> list[list[float]]:
        vectors: list[list[float] | None] = [None] * len(texts)
        missing: list[tuple[int, str, str]] = []
        with self._connect() as connection:
            for index, text in enumerate(texts):
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                cache_key = self._digest({"model": model, "text": text})
                row = connection.execute(
                    "SELECT vector_json FROM embeddings WHERE cache_key=? AND status='ok'",
                    (cache_key,),
                ).fetchone()
                if row:
                    vectors[index] = [float(value) for value in json.loads(row[0])]
                else:
                    missing.append((index, text, cache_key))

        batch_size = 64
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            batch_vectors: list[list[float]] | None = None
            last_error = ""
            for attempt in range(self.retries + 1):
                try:
                    batch_vectors = self._embed_many(
                        [text for _, text, _ in batch], model=model
                    )
                    if len(batch_vectors) != len(batch):
                        raise LLMError("embedding batch row count mismatch")
                    break
                except (LLMError, urllib.error.URLError) as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    if attempt >= self.retries:
                        break
                    time.sleep(min(15.0, 1.5 * (2**attempt)))
            if batch_vectors is None:
                raise LLMError(f"embedding batch failed: {last_error}")
            with self._write_lock, self._connect() as connection:
                for (index, text, cache_key), vector in zip(
                    batch, batch_vectors, strict=True
                ):
                    connection.execute(
                        """
                        INSERT OR REPLACE INTO embeddings
                        (cache_key, created_at, model, text_sha256, vector_json, status, error)
                        VALUES (?, ?, ?, ?, ?, 'ok', NULL)
                        """,
                        (
                            cache_key,
                            time.time(),
                            model,
                            hashlib.sha256(text.encode("utf-8")).hexdigest(),
                            json.dumps(vector),
                        ),
                    )
                    vectors[index] = vector
        return [vector for vector in vectors if vector is not None]

    def _embed_many(self, texts: Sequence[str], *, model: str) -> list[list[float]]:
        if not texts:
            return []
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            qualified = model if model.startswith("models/") else f"models/{model}"
            url = (
                "https://generativelanguage.googleapis.com/v1beta/"
                f"{qualified}:batchEmbedContents"
            )
            response = self._post(
                url,
                {
                    "requests": [
                        {
                            "model": qualified,
                            "content": {"parts": [{"text": text}]},
                        }
                        for text in texts
                    ]
                },
                {"x-goog-api-key": api_key},
                timeout=240,
            )
            rows = response.get("embeddings", [])
            vectors = [row.get("values", []) for row in rows]
        else:
            project = os.environ.get("GOOGLE_CLOUD_PROJECT") or self._gcloud_project()
            token = self._gcloud_token()
            location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
            host = f"{location}-aiplatform.googleapis.com"
            url = (
                f"https://{host}/v1/projects/{project}/locations/{location}/publishers/google/"
                f"models/{model}:predict"
            )
            response = self._post(
                url,
                {"instances": [{"content": text} for text in texts]},
                {"Authorization": f"Bearer {token}"},
                timeout=240,
            )
            vectors = [
                row.get("embeddings", {}).get("values", [])
                for row in response.get("predictions", [])
            ]
        if len(vectors) != len(texts) or any(not vector for vector in vectors):
            raise LLMError("embedding batch response is incomplete")
        return [[float(value) for value in vector] for vector in vectors]

    def _generate(self, model: str, payload: Mapping[str, Any], *, timeout: int) -> dict[str, Any]:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            qualified = model if model.startswith("models/") else f"models/{model}"
            url = f"https://generativelanguage.googleapis.com/v1beta/{qualified}:generateContent"
            return self._post(url, payload, {"x-goog-api-key": api_key}, timeout=timeout)

        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or self._gcloud_project()
        token = self._gcloud_token()
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
        url = (
            f"https://{host}/v1/projects/{project}/locations/{location}/publishers/google/"
            f"models/{model}:generateContent"
        )
        return self._post(url, payload, {"Authorization": f"Bearer {token}"}, timeout=timeout)

    def _embed_one(self, text: str, *, model: str) -> list[float]:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            qualified = model if model.startswith("models/") else f"models/{model}"
            url = f"https://generativelanguage.googleapis.com/v1beta/{qualified}:embedContent"
            payload = {"model": qualified, "content": {"parts": [{"text": text}]}}
            response = self._post(url, payload, {"x-goog-api-key": api_key}, timeout=180)
            values = response.get("embedding", {}).get("values", [])
        else:
            project = os.environ.get("GOOGLE_CLOUD_PROJECT") or self._gcloud_project()
            token = self._gcloud_token()
            location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
            host = f"{location}-aiplatform.googleapis.com"
            url = (
                f"https://{host}/v1/projects/{project}/locations/{location}/publishers/google/"
                f"models/{model}:predict"
            )
            response = self._post(
                url,
                {"instances": [{"content": text}], "parameters": {"outputDimensionality": 768}},
                {"Authorization": f"Bearer {token}"},
                timeout=180,
            )
            predictions = response.get("predictions", [])
            values = predictions[0].get("embeddings", {}).get("values", []) if predictions else []
        if not values:
            raise LLMError("embedding response contains no vector")
        return [float(value) for value in values]

    @staticmethod
    def _gcloud_project() -> str:
        completed = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True,
            text=True,
            check=False,
        )
        project = completed.stdout.strip()
        if completed.returncode != 0 or not project or project == "(unset)":
            raise LLMError("GOOGLE_CLOUD_PROJECT is not set and gcloud has no project")
        return project

    @staticmethod
    def _gcloud_token() -> str:
        completed = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True,
            text=True,
            check=False,
        )
        token = completed.stdout.strip()
        if completed.returncode != 0 or not token:
            detail = completed.stderr.strip().splitlines()[0] if completed.stderr.strip() else "unknown error"
            raise LLMError(f"gcloud authentication unavailable: {detail}")
        return token

    @staticmethod
    def _post(
        url: str,
        payload: Mapping[str, Any],
        extra_headers: Mapping[str, str],
        *,
        timeout: int,
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "natural-bcast/1.0", **extra_headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMError(f"HTTP {exc.code}: {detail[:1500]}") from exc
        if not isinstance(value, dict):
            raise LLMError("API response is not an object")
        return value

    @staticmethod
    def _extract_text(payload: Mapping[str, Any]) -> str:
        chunks = []
        for candidate in payload.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if isinstance(part.get("text"), str):
                    chunks.append(part["text"])
        text = "\n".join(chunks).strip()
        if not text:
            raise LLMError(f"response contained no text: {json.dumps(payload)[:1200]}")
        return text

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            value = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise
            value = json.loads(match.group(0))
        if not isinstance(value, dict):
            raise LLMError("parsed value is not an object")
        return value

    def usage_summary(self) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT model, purpose, status, usage_json FROM calls"
            ).fetchall()
            embedding_rows = connection.execute(
                "SELECT model, status, COUNT(*) FROM embeddings GROUP BY model, status"
            ).fetchall()
        result: dict[str, Any] = {"calls": len(rows), "by_model": {}, "by_purpose": {}, "embeddings": {}}
        for model, purpose, status, usage_json in rows:
            usage = json.loads(usage_json or "{}")
            for dimension, key in (("by_model", model), ("by_purpose", purpose)):
                item = result[dimension].setdefault(
                    key, {"calls": 0, "failed": 0, "prompt_tokens": 0, "output_tokens": 0}
                )
                item["calls"] += 1
                item["failed"] += int(status != "ok")
                item["prompt_tokens"] += int(usage.get("promptTokenCount", 0) or 0)
                item["output_tokens"] += int(usage.get("candidatesTokenCount", 0) or 0)
        for model, status, count in embedding_rows:
            result["embeddings"].setdefault(model, {})[status] = count
        return result
