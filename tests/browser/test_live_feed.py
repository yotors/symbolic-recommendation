"""Headless-Firefox proof that live feedback changes the rendered feed.

This test deliberately crosses every real boundary used by the demo:

    Firefox -> HTTP server -> live Lab -> PeTTaChainer -> browser DOM

It does not import the process-global fixture Lab and does not mock mining,
reasoning, HTTP, or JavaScript.  Screenshots and an audit JSON are written to
``/tmp/recommendation-live-feed-browser`` by default.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen

import pytest

selenium = pytest.importorskip("selenium")
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait


pytestmark = [
    pytest.mark.integration,
    pytest.mark.pettachainer,
    pytest.mark.slow,
]

WORKSPACE = Path(__file__).resolve().parents[3]
PYTHON = WORKSPACE / "PeTTaChainer" / ".venv" / "bin" / "python"
DEFAULT_GECKODRIVER = (
    Path.home() / ".cache" / "selenium" / "geckodriver" /
    "linux64" / "0.37.1" / "geckodriver"
)
ARTIFACT_DIR = Path(os.getenv(
    "RECOMMENDATION_BROWSER_ARTIFACT_DIR",
    "/tmp/recommendation-live-feed-browser",
))


def _free_loopback_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_json(
    url: str,
    process: subprocess.Popen[str] | None,
    timeout: float,
):
    deadline = time.monotonic() + timeout
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            raise AssertionError(
                f"recommendation server exited with {process.returncode}:\n{output}"
            )
        try:
            with urlopen(url, timeout=1) as response:
                return json.load(response)
        except (URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            time.sleep(0.2)
    raise AssertionError(f"server did not become ready: {last_error}")


def _visible_article_ids(driver: webdriver.Firefox) -> list[str]:
    ids = []
    for card in driver.find_elements(By.CSS_SELECTOR, "#feed article.card"):
        button = card.find_element(
            By.CSS_SELECTOR, 'button.feedback[data-action="skip"]'
        )
        if card.is_displayed():
            ids.append(str(button.get_attribute("data-article")))
    return ids


def _causal_payload(element) -> dict[str, object]:
    rule_ids = [
        value for value in str(element.get_attribute("data-rule-ids")).split(",")
        if value
    ]
    proof_text = str(element.get_attribute("data-proofs"))
    return {
        "article": str(element.get_attribute("data-article")),
        "before_rank": int(element.get_attribute("data-before-rank")),
        "after_rank": int(element.get_attribute("data-after-rank")),
        "before_score": float(element.get_attribute("data-before-score")),
        "after_score": float(element.get_attribute("data-after-score")),
        "before_ranking_score": float(
            element.get_attribute("data-before-ranking-score")
        ),
        "after_ranking_score": float(
            element.get_attribute("data-after-ranking-score")
        ),
        "feedback_evidence_changed": (
            element.get_attribute("data-feedback-evidence-changed") == "true"
        ),
        "before_feedback_signature": str(
            element.get_attribute("data-before-feedback-signature")
        ),
        "after_feedback_signature": str(
            element.get_attribute("data-after-feedback-signature")
        ),
        "score_method": str(element.get_attribute("data-score-method")),
        "rule_ids": rule_ids,
        "proof_text": proof_text,
        "proof_contains_feedback_rule": any(
            rule in proof_text for rule in rule_ids
        ),
    }


def _trace_revision(element) -> int:
    """Read new markup, with text fallback for an already-running server."""
    explicit = element.get_attribute("data-revision")
    if explicit:
        return int(explicit)
    match = re.search(r"Live reasoning update #(\d+)", element.text)
    return int(match.group(1)) if match else 0


def _geckodriver() -> Path:
    configured = os.getenv("GECKODRIVER")
    resolved = Path(configured) if configured else None
    if resolved and resolved.is_file():
        return resolved
    executable = shutil.which("geckodriver")
    if executable:
        return Path(executable)
    if DEFAULT_GECKODRIVER.is_file():
        return DEFAULT_GECKODRIVER
    pytest.skip("geckodriver is not installed or cached")


def test_skip_visibly_revises_the_real_petta_feed():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    external_url = os.getenv("RECOMMENDATION_BROWSER_BASE_URL", "").rstrip("/")
    process = None
    if external_url:
        base_url = external_url
    else:
        port = _free_loopback_port()
        base_url = f"http://127.0.0.1:{port}"
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            [str(PYTHON), "-m", "recommendation", "--fixture", "--port", str(port)],
            cwd=WORKSPACE,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    driver = None
    started = time.monotonic()
    audit: dict[str, object] = {
        "url": base_url,
        "server": ("existing real HTTP server" if external_url
                   else "real fixture HTTP server"),
        "browser": "headless Firefox",
        "reasoner": "PeTTaChainer",
    }
    try:
        initial_state = _wait_for_json(
            f"{base_url}/api/state", process, timeout=120
        )
        audit["initial_rule_version"] = initial_state["version"]
        audit["initial_worker_pid"] = initial_state["engine"]["worker_pid"]
        audit["dataset"] = initial_state.get("dataset", {}).get("name")

        options = Options()
        options.binary_location = shutil.which("firefox") or "/usr/bin/firefox"
        options.add_argument("-headless")
        # This development machine can have policy-installed browser add-ons
        # that block localhost inline scripts.  Safe mode gives the test a
        # deterministic, extension-free browser while retaining the real page,
        # HTTP requests, JavaScript and reasoner.
        options.add_argument("-safe-mode")
        options.set_preference("browser.cache.disk.enable", False)
        options.set_preference("browser.cache.memory.enable", False)
        service = Service(
            executable_path=str(_geckodriver()),
            log_output=str(ARTIFACT_DIR / "geckodriver.log"),
        )
        driver = webdriver.Firefox(options=options, service=service)
        driver.set_window_size(1280, 720)
        wait = WebDriverWait(driver, 90)
        driver.get(base_url)

        wait.until(lambda browser: (
            len(browser.find_elements(
                By.CSS_SELECTOR,
                '#feed button.feedback[data-action="skip"]',
            )) >= 5
            or browser.find_element(By.ID, "result").text.strip()
        ))
        startup_error = driver.find_element(By.ID, "result").text.strip()
        if startup_error:
            driver.save_screenshot(str(ARTIFACT_DIR / "startup-error.png"))
            (ARTIFACT_DIR / "startup-error.html").write_text(
                driver.page_source, encoding="utf-8"
            )
        assert not startup_error, f"browser initialization failed: {startup_error}"
        selected_user = str(driver.find_element(By.ID, "user").get_attribute("value"))
        if external_url and len(initial_state.get("users", [])) > 1:
            # Repeated verification must not inherit the default user's prior
            # test skips: those predicates are already in its pre-event score
            # and therefore cannot prove what this particular click caused.
            # Rotate from the end using the server's event count so consecutive
            # runs naturally exercise fresh MIND users without dataset IDs.
            users = initial_state["users"]
            online_events = int(
                (initial_state.get("engine") or {}).get("online_events") or 0
            )
            offset = online_events % (len(users) - 1) + 1
            selected_user = str(users[-offset]["id"])
            driver.execute_script(
                "const u=document.getElementById('user');"
                "u.value=arguments[0];"
                "u.dispatchEvent(new Event('change',{bubbles:true}));",
                selected_user,
            )
            wait.until(lambda browser: (
                browser.find_element(By.ID, "user").get_attribute("value")
                    == selected_user
                and len(browser.find_elements(
                    By.CSS_SELECTOR,
                    '#feed button.feedback[data-action="skip"]',
                )) >= 5
            ))
        audit["user"] = selected_user
        before = _visible_article_ids(driver)
        # The short viewport keeps the sentinel below the first five cards, so
        # the fixture retains a real unserved queue for the interaction.
        assert len(before) == 5, (
            "infinite-scroll consumed the fixture before feedback; "
            f"visible IDs were {before}"
        )
        before_path = ARTIFACT_DIR / "before-skip.png"
        driver.save_screenshot(str(before_path))

        # Keep the fixture's deterministic climate-sibling check exact. A real
        # MIND page can start with an article whose taxonomy has no sibling in
        # the current unserved window, so try bounded successive *displayed*
        # cards until one newly causes proof-backed negative evidence.
        max_attempts = 8 if external_url else 1
        current_before = before
        attempted: list[str] = []
        causal_data: dict[str, object] | None = None
        causal_skip: str | None = None
        causal_revision: int | None = None
        causal_trace_text: str | None = None
        trace = None
        final_revision = 0
        for _attempt in range(max_attempts):
            cards = wait.until(lambda browser: (
                found if (found := browser.find_elements(
                    By.CSS_SELECTOR, "#feed article.card"
                )) else False
            ))
            feedback_card = cards[0 if external_url else -1]
            skip = feedback_card.find_element(
                By.CSS_SELECTOR, 'button.feedback[data-action="skip"]'
            )
            skipped = str(skip.get_attribute("data-article"))
            attempted.append(skipped)
            previous_revision = final_revision
            driver.execute_script("arguments[0].click()", skip)

            trace = wait.until(lambda browser: (
                element
                if (element := browser.find_element(By.ID, "feedbackTrace")).is_displayed()
                and "PeTTaChainer reranked" in element.text
                and _trace_revision(element) > previous_revision
                else False
            ))
            final_revision = _trace_revision(trace)
            wait.until(lambda browser: (
                (ids := _visible_article_ids(browser)) and ids != current_before
            ))
            current_after = _visible_article_ids(driver)
            causal_elements = driver.find_elements(
                By.CSS_SELECTOR, "#feedbackTrace .causal-demotion"
            )
            if causal_elements:
                candidate = _causal_payload(causal_elements[0])
                score_decreased = candidate["after_score"] < candidate["before_score"]
                ranking_score_decreased = (
                    candidate["after_ranking_score"]
                    < candidate["before_ranking_score"]
                )
                if (candidate["feedback_evidence_changed"]
                        and (score_decreased or ranking_score_decreased)):
                    causal_data = candidate
                    causal_skip = skipped
                    causal_revision = final_revision
                    causal_trace_text = trace.text
                    break
            current_before = current_after

        assert causal_data is not None, (
            f"no causal negative PeTTa proof after skips {attempted}"
        )
        assert trace is not None
        assert causal_revision is not None
        assert causal_trace_text is not None
        after = _visible_article_ids(driver)
        after_path = ARTIFACT_DIR / "after-skip.png"
        driver.save_screenshot(str(after_path))

        assert causal_skip not in after
        assert after != before
        if not external_url:
            assert final_revision == 1
        assert "PeTTaChainer reranked" in causal_trace_text
        assert "queued articles" in causal_trace_text
        assert (
            "0 candidates received new/changed negative proof evidence"
            not in causal_trace_text
        )

        rank_demoted = causal_data["after_rank"] > causal_data["before_rank"]
        score_decreased = causal_data["after_score"] < causal_data["before_score"]
        ranking_score_decreased = (
            causal_data["after_ranking_score"]
            < causal_data["before_ranking_score"]
        )
        assert causal_data["feedback_evidence_changed"]
        assert (causal_data["before_feedback_signature"]
                != causal_data["after_feedback_signature"])
        assert score_decreased or ranking_score_decreased, (
            "negative PeTTa evidence changed neither own score: "
            f"{causal_data}"
        )
        # The deterministic fixture must retain the stronger known result.
        if not external_url:
            assert rank_demoted and score_decreased
            assert "0 positions changed" not in causal_trace_text
        assert causal_data["score_method"] == "pettachainer_live_feedback_revision"
        assert any(
            rule.startswith("feedback_skip_")
            for rule in causal_data["rule_ids"]
        )
        assert causal_data["proof_contains_feedback_rule"], (
            f"feedback rules were absent from proof: {causal_data}"
        )

        post_feedback_cards = driver.find_elements(
            By.CSS_SELECTOR, "#feed article.card"
        )
        assert post_feedback_cards
        assert all(
            card.get_attribute("data-queue-revision") == str(final_revision)
            for card in post_feedback_cards
        ), "a pre-feedback infinite-scroll response repopulated the revised feed"
        assert not driver.find_element(By.ID, "result").text.strip(), (
            "browser reported an API/JavaScript failure: "
            f"{driver.find_element(By.ID, 'result').text}"
        )

        audit.update({
            "skipped_article": causal_skip,
            "skipped_articles_attempted": attempted,
            "feedback_attempts": len(attempted),
            "final_queue_revision": final_revision,
            "causal_queue_revision": causal_revision,
            "before_visible_article_ids": before,
            "after_visible_article_ids": after,
            "feed_changed": before != after,
            "causal_demotion": {
                key: value for key, value in causal_data.items()
                if key != "proof_text"
            },
            "feedback_trace": causal_trace_text,
            "final_feedback_trace": trace.text,
            "before_screenshot": str(before_path),
            "after_screenshot": str(after_path),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        (ARTIFACT_DIR / "result.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        if driver is not None:
            driver.quit()
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
