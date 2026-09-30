"""Parse fpMiner and PeTTa target-rule output into ranking rules."""
from __future__ import annotations

import re


POSITIVE = {"click", "like", "complete"}
FEATURES = (
    "topic", "subcategory", "format", "affinity", "affinity_level",
    "recent_affinity", "long_affinity", "history_size_bucket",
    "entity_overlap", "entity_overlap_detail", "history_topic_count_bucket",
    "recent_topic_count_bucket", "topic_rank_bucket", "subcategory_affinity",
    "topic_recency_bucket", "subcategory_recency_bucket",
    "time_bucket", "ctr_bucket", "freshness_bucket", "position_bucket",
    "title_overlap_detail", "rel_entity_continuity_scope",
    "rel_concept_continuity_scope", "mi_topic_candidate_match_rank",
    "mi_subcategory_candidate_match_rank", "mi_entity_candidate_match_rank",
)
STV_RE = re.compile(r"\(STV\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\)")


def balanced_forms(text, head="supportOf"):
    forms = []
    for start in (match.start() for match in
                  re.finditer(r"\(" + head + r"\b", text)):
        depth = 0; quoted = False; escaped = False
        for index, character in enumerate(text[start:], start):
            if quoted:
                if escaped: escaped = False
                elif character == "\\": escaped = True
                elif character == '"': quoted = False
            elif character == '"': quoted = True
            elif character == "(": depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    forms.append(text[start:index + 1]); break
    return forms


def parse_rules(raw, limit=None, features=FEATURES):
    clause_re = re.compile(
        r'\((' + "|".join((*features, "engagement")) + r')\s+[^\s()]+\s+"([^"]+)"\)'
    )
    unique = {}
    for form in balanced_forms(" ".join(raw)):
        clauses = clause_re.findall(form)
        target = next((v for p, v in clauses if p == "engagement"), None)
        premises = tuple((p, v) for p, v in clauses if p in features)
        tvs = STV_RE.findall(form); support = re.search(r"\)\s+(\d+)\)$", form)
        if target in POSITIVE and premises and tvs and support:
            strength, confidence = map(float, tvs[0])
            negative_strength,negative_confidence=(
                map(float, tvs[1]) if len(tvs) > 1 else (0.0, 0.0)
            )
            unique[(premises, target)] = {
                "premises":premises, "target":target,
                "support":int(support.group(1)),
                "strength":strength, "confidence":confidence,
                "discovery_ctv":{
                    "positive":{"strength":strength,"confidence":confidence},
                    "negative":{
                        "strength":negative_strength,
                        "confidence":negative_confidence,
                    },
                    "complete":len(tvs) > 1,
                },
            }
    rules = sorted(unique.values(),
                   key=lambda rule:(-rule["support"],-rule["strength"],rule["premises"]))
    if limit is not None: rules = rules[:limit]
    for index, rule in enumerate(rules, 1):
        rule.update(id=f"mined_{index}",
                    source="recommendation/miner/fpMiner.metta")
    return rules


def parse_petta_target_rules(raw, *, features=FEATURES):
    """Decode target-aware rule atoms without recomputing their statistics."""
    clause_re = re.compile(
        r'\((' + "|".join((*features, "engagement"))
        + r')\s+[^\s()]+\s+"([^"]+)"\)'
    )
    metric_patterns = {
        "target_auc": r"\(AUC\s+([0-9.eE+-]+)\)",
        "target_auc_gain": r"\(AUC-Gain\s+([0-9.eE+-]+)\)",
        "target_youden_j": r"\(Youden-J\s+([0-9.eE+-]+)\)",
        "target_wracc": r"\(WRAcc\s+([0-9.eE+-]+)\)",
        "target_information_gain": r"\(Information-Gain\s+([0-9.eE+-]+)\)",
        "target_log_odds": r"\(Log-Odds\s+([0-9.eE+-]+)\)",
    }
    optional_metric_patterns = {
        "target_parent_precision": r"\(Parent-Precision\s+([0-9.eE+-]+)\)",
        "target_incremental_precision": r"\(Incremental-Precision\s+([0-9.eE+-]+)\)",
        "target_incremental_wracc": r"\(Incremental-WRAcc\s+([0-9.eE+-]+)\)",
        "target_mdl_gain": r"\(MDL-Gain\s+([0-9.eE+-]+)\)",
        "target_hierarchical_parent_precision":
            r"\(Hierarchical-Parent-Precision\s+([0-9.eE+-]+)\)",
        "target_hierarchical_precision":
            r"\(Hierarchical-Precision\s+([0-9.eE+-]+)\)",
    }
    unique = {}
    text = " ".join(str(value) for value in raw)
    for form in balanced_forms(text, "targetScoreOf"):
        clauses = clause_re.findall(form)
        target = next((value for predicate, value in clauses
                       if predicate == "engagement"), None)
        premises = tuple((predicate, value) for predicate, value in clauses
                         if predicate in features)
        stvs = STV_RE.findall(form)
        support_match = re.search(r"\)\s+([0-9]+)\s*\)$", form)
        contingency_match = re.search(
            r"\(Contingency\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\)",
            form,
        )
        metrics = {
            name: float(match.group(1))
            for name, pattern in metric_patterns.items()
            if (match := re.search(pattern, form)) is not None
        }
        optional_metrics = {
            name: float(match.group(1))
            for name, pattern in optional_metric_patterns.items()
            if (match := re.search(pattern, form)) is not None
        }
        if (target not in POSITIVE or not premises or len(stvs) < 2
                or support_match is None or contingency_match is None
                or len(metrics) != len(metric_patterns)):
            continue
        strength, confidence = map(float, stvs[0])
        negative_strength, negative_confidence = map(float, stvs[1])
        contingency = tuple(map(int, contingency_match.groups()))
        key = (premises, target)
        record = {
            "premises": premises,
            "target": target,
            "support": int(support_match.group(1)),
            "strength": strength,
            "confidence": confidence,
            "negative_strength": negative_strength,
            "negative_confidence": negative_confidence,
            "target_contingency": {
                "n11": contingency[0], "n10": contingency[1],
                "n01": contingency[2], "n00": contingency[3],
            },
            "discovery_ctv": {
                "positive": {"strength": strength, "confidence": confidence},
                "negative": {
                    "strength": negative_strength,
                    "confidence": negative_confidence,
                },
                "complete": True,
            },
            "petta_target_aware": True,
            "source": "recommendation/miner/fpMiner.metta#target-aware",
            **metrics,
            **optional_metrics,
        }
        previous = unique.get(key)
        if previous is not None and previous != record:
            raise ValueError("PeTTa target miner emitted conflicting duplicate rules")
        unique[key] = record
    rules = [unique[key] for key in sorted(unique)]
    for index, rule in enumerate(rules, 1):
        rule["id"] = f"petta_target_{index}"
    return rules


def proof_tv(proof):
    values = STV_RE.findall(proof)
    return tuple(map(float, values[-1])) if values else (0.0, 0.0)
