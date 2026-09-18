"""HTTP boundary for the real PeTTaChainer semantic and reasoning services.

The live lab is intentionally runnable without ``python-dotenv``.  Values in
``recommendation/.env`` are treated as local defaults; an exported environment
variable always wins.  Credentials are resolved when a client is constructed,
kept only on that client, and never returned by this module.
"""
from __future__ import annotations

import ipaddress
import json
import math
import os
import re
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen


from ..paths import ENV_FILE, WORKSPACE_ROOT


_LOCAL_ENV = ENV_FILE
_STACK_ENV = WORKSPACE_ROOT / "PeTTaChainer" / ".env"

# These limits mirror the PeTTaChainer request contract where possible and add
# a smaller client-side response envelope for an interactive preview.  They
# prevent an accidentally unbounded model response from being retained by the
# lab process.
MIN_TIMEOUT_SECONDS = 0.1
MAX_TIMEOUT_SECONDS = 300.0
MAX_SEMANTIC_INPUT_CHARS = 10_000
MAX_SEMANTIC_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_SEMANTIC_STATEMENTS = 256
MAX_STATEMENT_CHARS = 100_000
MAX_PREDICATE_SCHEMA_ITEMS = 100
EXPECTED_NL2PLN_CONTRACT = "pettachainer.horn.v1"

# A deliberately bounded, dataset-independent projection.  NL2PLN may choose
# values from the article text, but it may not invent new predicate *shapes*
# for the miner on every request.
RECOMMENDATION_PREDICATE_SCHEMA = (
    {"name": "HasTopic", "arity": 2,
     "description": "article and a normalized subject topic"},
    {"name": "MentionsActor", "arity": 2,
     "description": "article and a person, team, organization, or place mentioned"},
    {"name": "DescribesEvent", "arity": 3,
     "description": "article, normalized event type, and principal actor"},
    {"name": "HasTone", "arity": 2,
     "description": "article and normalized narrative tone"},
    {"name": "ExpressesStance", "arity": 3,
     "description": "article, proposition or actor, and normalized stance"},
    {"name": "AboutStory", "arity": 2,
     "description": "article and a normalized continuing-story identifier"},
)


class PeTTaChainerClientError(RuntimeError):
    """Base class for failures at the optional semantic HTTP boundary."""


class PeTTaChainerConfigurationError(PeTTaChainerClientError, ValueError):
    """The client cannot make a request with its current configuration."""


class PeTTaChainerInputError(PeTTaChainerClientError, ValueError):
    """The caller supplied input outside the bounded semantic contract."""


class PeTTaChainerUpstreamError(PeTTaChainerClientError):
    """PeTTaChainer rejected the request or could not be reached."""


class PeTTaChainerTimeoutError(PeTTaChainerUpstreamError, TimeoutError):
    """The bounded PeTTaChainer request timed out."""


class PeTTaChainerProtocolError(PeTTaChainerClientError):
    """PeTTaChainer returned data outside the expected response contract."""


def _dotenv_setting(path, names):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    wanted = set(names)
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() in wanted:
            value = value.strip().strip('"').strip("'")
            if value:
                return value
    return ""


def _setting(*names, default=""):
    """Resolve an exported setting, then a task-local dotenv value."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return _dotenv_setting(_LOCAL_ENV, names) or default


def _local_stack_api_secret():
    """Resolve the compose API secret for the loopback development stack.

    Compose stores inbound keys as ``owner:secret`` entries while this client
    sends only the secret as its bearer token.  Reading the sibling file avoids
    duplicating a long-lived secret into a second dotenv file.  This fallback
    is used only for a loopback URL; exported configuration remains preferred.
    """
    entries = _dotenv_setting(_STACK_ENV, ("PETTACHAINER_API_KEYS",))
    first = entries.split(",", 1)[0]
    return first.split(":", 1)[1] if ":" in first else ""


_PREDICATE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,127}$")


def _normalize_base_url(value):
    if not isinstance(value, str) or not value:
        raise PeTTaChainerConfigurationError("PETTACHAINER_URL must be a non-empty URL")
    if value != value.strip() or any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise PeTTaChainerConfigurationError("PETTACHAINER_URL contains invalid whitespace")
    try:
        parsed = urlsplit(value)
        # Accessing ``port`` makes urllib validate malformed and out-of-range
        # ports instead of deferring that ambiguity until request time.
        parsed.port
    except ValueError as exc:
        raise PeTTaChainerConfigurationError("PETTACHAINER_URL is invalid") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc or not parsed.hostname:
        raise PeTTaChainerConfigurationError("PETTACHAINER_URL must use http or https")
    # A bearer-token origin is never allowed to contain userinfo.  In
    # particular, ``localhost@remote.example`` must not resemble loopback to a
    # human while urllib correctly sends the request to the remote hostname.
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        raise PeTTaChainerConfigurationError("PETTACHAINER_URL must not contain userinfo")
    if parsed.query or parsed.fragment:
        raise PeTTaChainerConfigurationError("PETTACHAINER_URL must not contain a query or fragment")
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, path, "", ""))


def _is_loopback_url(base_url):
    """Classify the parsed hostname, never a string prefix, as loopback."""
    hostname = urlsplit(base_url).hostname
    if not hostname:
        return False
    hostname = hostname.lower()
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _bounded_timeout(value):
    if isinstance(value, bool):
        raise PeTTaChainerConfigurationError("semantic timeout must be a number of seconds")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise PeTTaChainerConfigurationError("semantic timeout must be a number of seconds") from exc
    if not math.isfinite(resolved) or not MIN_TIMEOUT_SECONDS <= resolved <= MAX_TIMEOUT_SECONDS:
        raise PeTTaChainerConfigurationError(
            f"semantic timeout must be finite and between {MIN_TIMEOUT_SECONDS:g} and "
            f"{MAX_TIMEOUT_SECONDS:g} seconds"
        )
    return resolved


def _bounded_text(value, label):
    if not isinstance(value, str):
        raise PeTTaChainerInputError(f"{label} must be text")
    value = value.strip()
    if not value:
        raise PeTTaChainerInputError(f"{label} cannot be empty")
    if len(value) > MAX_SEMANTIC_INPUT_CHARS:
        raise PeTTaChainerInputError(
            f"{label} exceeds the {MAX_SEMANTIC_INPUT_CHARS:,}-character semantic input limit"
        )
    return value


def _bounded_predicate_schema(predicate_schema):
    if not isinstance(predicate_schema, (list, tuple)):
        raise PeTTaChainerInputError("predicate_schema must be a list or tuple")
    if len(predicate_schema) > MAX_PREDICATE_SCHEMA_ITEMS:
        raise PeTTaChainerInputError(
            f"predicate_schema exceeds the {MAX_PREDICATE_SCHEMA_ITEMS}-item limit"
        )
    normalized = []
    names = set()
    for item in predicate_schema:
        if not isinstance(item, dict):
            raise PeTTaChainerInputError("predicate_schema entries must be objects")
        name = item.get("name")
        arity = item.get("arity")
        description = item.get("description")
        if not isinstance(name, str) or not _PREDICATE_NAME.fullmatch(name):
            raise PeTTaChainerInputError("predicate_schema contains an invalid predicate name")
        if name in names:
            raise PeTTaChainerInputError("predicate_schema contains duplicate predicate names")
        if isinstance(arity, bool) or not isinstance(arity, int) or not 0 <= arity <= 16:
            raise PeTTaChainerInputError("predicate_schema contains an invalid arity")
        if not isinstance(description, str) or not 1 <= len(description) <= 500:
            raise PeTTaChainerInputError("predicate_schema contains an invalid description")
        names.add(name)
        normalized.append({"name": name, "arity": arity, "description": description})
    return normalized


def _validate_converter(converter):
    if not isinstance(converter, dict):
        raise PeTTaChainerProtocolError("semantic response has invalid converter provenance")
    for field, maximum in (("contract_version", 128), ("model", 256), ("namespace", 256)):
        value = converter.get(field)
        if not isinstance(value, str) or not 1 <= len(value) <= maximum:
            raise PeTTaChainerProtocolError(
                f"semantic response is missing valid converter {field} provenance"
            )
    if converter["contract_version"] != EXPECTED_NL2PLN_CONTRACT:
        raise PeTTaChainerProtocolError(
            f"unsupported NL2PLN contract: {converter['contract_version']}"
        )
    warnings = converter.get("warnings", [])
    if (not isinstance(warnings, list) or len(warnings) > 100
            or any(not isinstance(item, str) or len(item) > 1_000 for item in warnings)):
        raise PeTTaChainerProtocolError("semantic response has invalid converter warnings")
    if "cached" in converter and not isinstance(converter["cached"], bool):
        raise PeTTaChainerProtocolError("semantic response has invalid converter cache provenance")
    attempts = converter.get("attempts")
    if (attempts is not None
            and (isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 1_000)):
        raise PeTTaChainerProtocolError("semantic response has invalid converter attempt provenance")
    if "usage" in converter and not isinstance(converter["usage"], dict):
        raise PeTTaChainerProtocolError("semantic response has invalid converter usage provenance")
    return dict(converter)


def _validate_preview_response(result, knowledge_base):
    if result.get("conversion") != "nl2pln":
        raise PeTTaChainerProtocolError("semantic preview did not use nl2pln conversion")
    if result.get("committed") is not False:
        raise PeTTaChainerProtocolError("semantic preview response is not explicitly non-committing")
    response_kb = result.get("knowledge_base_id")
    if response_kb is None:
        raise PeTTaChainerProtocolError(
            "semantic preview response is missing its knowledge base identity"
        )
    if str(response_kb) != str(knowledge_base):
        raise PeTTaChainerProtocolError("semantic preview response belongs to another knowledge base")
    statements = result.get("statements")
    if not isinstance(statements, list):
        raise PeTTaChainerProtocolError("semantic preview statements must be a list")
    if len(statements) > MAX_SEMANTIC_STATEMENTS:
        raise PeTTaChainerProtocolError(
            f"semantic preview exceeds the {MAX_SEMANTIC_STATEMENTS}-statement limit"
        )
    declared_count = result.get("statement_count")
    if (declared_count is not None
            and (isinstance(declared_count, bool) or not isinstance(declared_count, int)
                 or declared_count != len(statements))):
        raise PeTTaChainerProtocolError("semantic preview statement count does not match its payload")
    sources = []
    for item in statements:
        if not isinstance(item, dict):
            raise PeTTaChainerProtocolError("semantic preview contains an invalid statement object")
        source = item.get("source")
        if (not isinstance(source, str) or not source.strip()
                or len(source) > MAX_STATEMENT_CHARS):
            raise PeTTaChainerProtocolError("semantic preview contains an invalid statement source")
        idempotency_key = item.get("idempotency_key")
        if idempotency_key is not None and (
            not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 200
        ):
            raise PeTTaChainerProtocolError("semantic preview contains an invalid idempotency key")
        mine_patterns = item.get("mine_patterns")
        if mine_patterns is not None and not isinstance(mine_patterns, bool):
            raise PeTTaChainerProtocolError("semantic preview contains an invalid mining flag")
        sources.append(source)
    if "converter" not in result:
        raise PeTTaChainerProtocolError(
            "semantic preview response is missing converter provenance"
        )
    converter = _validate_converter(result["converter"])
    return sources, converter


class PeTTaChainerClient:
    def __init__(self, base_url=None, api_key=None, knowledge_base=None, timeout=None):
        configured_url = base_url if base_url is not None else _setting(
            "PETTACHAINER_URL", "PETTACHAINER_BASE_URL",
            default="http://127.0.0.1:8000",
        )
        self.base_url = _normalize_base_url(configured_url)
        exported_key = os.getenv("PETTACHAINER_API_KEY", "")
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = (exported_key
                            or (_local_stack_api_secret() if _is_loopback_url(self.base_url) else "")
                            or _setting("PETTACHAINER_API_KEY"))
        self.knowledge_base = (
            knowledge_base if knowledge_base is not None else _setting("PETTACHAINER_KB_ID")
        )
        configured_timeout = _setting(
            "PETTACHAINER_SEMANTIC_TIMEOUT_SECONDS",
            "NL2PLN_TIMEOUT_SECONDS",
            default="90",
        )
        self.timeout = _bounded_timeout(timeout if timeout is not None else configured_timeout)
        # Cache invalidation is deliberately explicit. Deployments should bump
        # PETTACHAINER_SEMANTIC_CACHE_VERSION with a prompt/ontology/model
        # change; the lab also applies a short TTL as a second line of defense.
        self.semantic_cache_version = _setting(
            "PETTACHAINER_SEMANTIC_CACHE_VERSION", "NL2PLN_MODEL",
            default="unversioned",
        )

    def _require_configured(self):
        if not isinstance(self.api_key, str) or not self.api_key:
            raise PeTTaChainerConfigurationError("PETTACHAINER_API_KEY is required")
        if not isinstance(self.knowledge_base, str) or not self.knowledge_base:
            raise PeTTaChainerConfigurationError("PETTACHAINER_KB_ID is required")

    def _post(self, path, payload):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            self.base_url + path,
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read(MAX_SEMANTIC_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            raise PeTTaChainerUpstreamError(
                f"semantic service rejected the request with HTTP {exc.code}"
            ) from exc
        except URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise PeTTaChainerTimeoutError("semantic service request timed out") from exc
            raise PeTTaChainerUpstreamError("semantic service is unavailable") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise PeTTaChainerTimeoutError("semantic service request timed out") from exc
        except OSError as exc:
            raise PeTTaChainerUpstreamError("semantic service is unavailable") from exc
        if len(raw) > MAX_SEMANTIC_RESPONSE_BYTES:
            raise PeTTaChainerProtocolError(
                f"semantic response exceeds the {MAX_SEMANTIC_RESPONSE_BYTES:,}-byte limit"
            )
        try:
            result = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise PeTTaChainerProtocolError("semantic service returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise PeTTaChainerProtocolError("semantic service response must be a JSON object")
        return result

    def parse_text(self, text, predicate_schema=()):
        """Use PeTTaChainer's configured NL2PLN service, not local heuristics."""
        self._require_configured()
        text = _bounded_text(text, "semantic text")
        schema = _bounded_predicate_schema(predicate_schema)
        result = self._post(f"/v1/knowledge-bases/{quote(self.knowledge_base, safe='')}/knowledge", {
            "text": text,
            "conversion": "nl2pln",
            "predicate_schema": schema,
            "preview": True,
        })
        _validate_preview_response(result, self.knowledge_base)
        return result

    def parse_article(self, article):
        """Preview bounded recommendation semantics for one article.

        This is a real remote conversion through PeTTaChainer's private NL2PLN
        connection.  Preview mode is non-mutating; a later ingestion job can
        cache and commit approved projections by content hash.
        """
        title_value = article.get("title", "")
        abstract_value = article.get("abstract", "")
        title = "" if title_value is None else str(title_value).strip()
        abstract = "" if abstract_value is None else str(abstract_value).strip()
        if not title and not abstract:
            raise PeTTaChainerInputError("article has no title or abstract to parse")
        text = "\n".join(part for part in (
            f"Title: {title}" if title else "",
            f"Abstract: {abstract}" if abstract else "",
        ) if part)
        result = self.parse_text(text, RECOMMENDATION_PREDICATE_SCHEMA)
        statements, converter = _validate_preview_response(result, self.knowledge_base)
        provenance = {
            key: converter[key]
            for key in ("contract_version", "model", "namespace")
            if converter is not None
        }
        return {
            "article_id": str(article.get("id", "")),
            "conversion": "nl2pln",
            "committed": False,
            "statement_count": len(statements),
            "statements": statements,
            "converter": converter or {},
            "converter_provenance": provenance,
            "predicate_schema": list(RECOMMENDATION_PREDICATE_SCHEMA),
        }

    def query(self, question, predicate_schema=()):
        self._require_configured()
        question = _bounded_text(question, "semantic question")
        schema = _bounded_predicate_schema(predicate_schema)
        return self._post(f"/v1/knowledge-bases/{quote(self.knowledge_base, safe='')}/query", {
            "question": question,
            "predicate_schema": schema,
        })
