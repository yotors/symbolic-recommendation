"""Live feed, feature grounding and semantic-preview methods."""
from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import time
import uuid

from ..adapters.mind import (
    history_feature_context, prepare_history_feature_workspace,
    subcategory_transition_score,
)
from ..features.lexical_workspace import build_lexical_workspace_facts
from ..features.llm_workspace import LLM_WORKSPACE_FEATURES, build_llm_workspace_facts
from ..features.recency_workspace import (
    RECENCY_WORKSPACE_FEATURES, build_recency_workspace_facts,
)
from ..features.relational_workspace import (
    REL_CONCEPT_CONTINUITY_SCOPE, REL_ENTITY_CONTINUITY_SCOPE,
    build_relational_plans, reduce_concept_relational_proofs,
    reduce_relational_proofs,
)
from ..features.semantic_workspace import build_semantic_workspace_facts
from ..integrations.engine import (
    EXPECTED_NL2PLN_CONTRACT, RECOMMENDATION_PREDICATE_SCHEMA,
    PeTTaChainerConfigurationError,
)
from ..mining.rule_parser import POSITIVE, proof_tv


class SemanticPreviewBusyError(RuntimeError):
    """The one bounded semantic conversion slot is already occupied."""


def make_serving_mixin(
    *,
    CONTEXT_FEATURES,
    DEFAULT_MAX_PROOF_CACHE_ENTRIES,
    FEATURE_PROFILES,
    FEED_DELIVERY_HISTORY_LIMIT,
    FEED_DELIVERY_ROW_LIMIT,
    LIVE_NEGATIVE_FEATURE,
    LIVE_NEGATIVE_GENERALIZATION_WINDOW,
    LIVE_NEGATIVE_RULE_IDS,
    RELATIONAL_LIVE_HISTORY_LIMIT,
    RELATIONAL_PROOF_FIELDS,
    RELATIONAL_QUERY_STEPS_PER_ROOT,
    SEMANTIC_CACHE_MAX_BYTES,
    SEMANTIC_CACHE_MAX_ENTRIES,
    SEMANTIC_CACHE_TTL_SECONDS,
    SEMANTIC_SINGLE_FLIGHT_WAIT_SECONDS,
):
    class ServingMixin:
        def article(self, aid):
            try: return self._articles[str(aid)]
            except KeyError as exc: raise ValueError(f"unknown article: {aid}") from exc

        @staticmethod
        def _presentation_from_annotation(annotation):
            """Return bounded, human-readable semantic labels for feed cards."""
            if not isinstance(annotation,dict):
                return {}
            canonical=(annotation.get("provenance",{}).get("canonicalization",{})
                       .get("mappings",{}))

            def labels(field,limit):
                result=[]
                for item in canonical.get(field,[]) or []:
                    value=(item.get("lexical") if isinstance(item,dict) else None)
                    if value and value not in result:
                        result.append(str(value))
                    if len(result)>=limit:
                        break
                if result:
                    return result
                raw_values=annotation.get(field,[]) or []
                if isinstance(raw_values,str):
                    raw_values=[raw_values]
                for value in raw_values:
                    # Portable IDs retain a readable slug before the content hash.
                    text=str(value).split(":",1)[-1].split("~",1)[0]
                    text=text.replace("-"," ").strip()
                    if text and text not in result:
                        result.append(text)
                    if len(result)>=limit:
                        break
                return result

            formats=labels("format",1)
            if not formats and annotation.get("format"):
                value=str(annotation["format"]).split(":",1)[-1].split("~",1)[0]
                formats=[value.replace("-"," ")]
            return {
                "concepts":labels("concepts",5),
                "audiences":labels("audiences",3),
                "events":labels("event_types",2),
                "intents":labels("intents",2),
                "semantic_format":formats[0] if formats else None,
            }

        def public_article(self, aid):
            """Article content plus non-ranking metadata safe for the browser."""
            article=dict(self.article(aid))
            presentation=self._article_presentation.get(str(aid))
            if presentation:
                article["presentation"]=presentation
            return article

        def preview_semantics(self, article):
            """Parse immutable content through one bounded, versioned preview slot."""
            if self.symbolic_only:
                raise PeTTaChainerConfigurationError("NL2PLN is disabled in symbolic-only mode")
            if self.semantic_client is None:
                raise PeTTaChainerConfigurationError(
                    "semantic parser configuration is invalid"
                )
            article_id=str(article.get("id",""))
            payload=json.dumps({
                "title":str(article.get("title","")),
                "abstract":str(article.get("abstract","")),
                "base_url":self.semantic_client.base_url,
                "knowledge_base":self.semantic_client.knowledge_base,
                "predicate_schema":RECOMMENDATION_PREDICATE_SCHEMA,
                "contract":EXPECTED_NL2PLN_CONTRACT,
                "semantic_cache_version":self.semantic_client.semantic_cache_version,
            },ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")
            key=hashlib.sha256(payload).hexdigest()
            # The semantic lock is deliberately separate from ``Lab.lock``. It is
            # a single-flight/model-quota boundary, but callers wait at most one
            # second rather than consuming unbounded HTTP threads behind a slow
            # provider request.
            if not self._semantic_lock.acquire(timeout=SEMANTIC_SINGLE_FLIGHT_WAIT_SECONDS):
                raise SemanticPreviewBusyError("semantic preview is busy")
            try:
                now=time.monotonic()
                while self._semantic_preview_cache:
                    oldest_key,oldest=next(iter(self._semantic_preview_cache.items()))
                    if oldest["expires_at"]>now:
                        break
                    self._semantic_preview_cache.pop(oldest_key)
                    self._semantic_preview_cache_bytes-=oldest["bytes"]
                cached=self._semantic_preview_cache.get(key)
                if cached is not None:
                    if cached["expires_at"]<=now:
                        self._semantic_preview_cache.pop(key)
                        self._semantic_preview_cache_bytes-=cached["bytes"]
                    else:
                        self._semantic_preview_cache.move_to_end(key)
                        return {**cached["result"],"article_id":article_id,
                                "cached":True,"content_hash":key,
                                "lab_instance_id":self.instance_id}
                result={**self.semantic_client.parse_article(article),"cached":False,
                        "content_hash":key,"lab_instance_id":self.instance_id}
                cache_result={**result,"cached":False}
                cache_bytes=len(json.dumps(
                    cache_result,ensure_ascii=False,separators=(",",":")
                ).encode("utf-8"))
                if cache_bytes<=SEMANTIC_CACHE_MAX_BYTES:
                    self._semantic_preview_cache[key]={
                        "result":cache_result,"bytes":cache_bytes,
                        "expires_at":now+SEMANTIC_CACHE_TTL_SECONDS,
                    }
                    self._semantic_preview_cache_bytes+=cache_bytes
                    while (len(self._semantic_preview_cache)>SEMANTIC_CACHE_MAX_ENTRIES
                           or self._semantic_preview_cache_bytes>SEMANTIC_CACHE_MAX_BYTES):
                        _discarded_key,discarded=self._semantic_preview_cache.popitem(last=False)
                        self._semantic_preview_cache_bytes-=discarded["bytes"]
                return result
            finally:
                self._semantic_lock.release()
        def user_topics(self,user):
            try: profile=self.data["users"][user]
            except KeyError as exc: raise ValueError(f"unknown user: {user}") from exc
            if isinstance(profile,dict): return profile.get("topics",profile.get("interests",[]))
            return profile

        def _mutable_user_profile(self,user):
            """Return a mutable live profile without requiring one dataset schema.

            Small fixtures historically store a bare topic list, while MIND
            adapters store a mapping with causal history.  Normalizing only when
            the user actually sends feedback keeps both input contracts valid and
            gives live negative evidence one explicit, bounded home.
            """
            profile=self.data["users"][user]
            if isinstance(profile,dict): return profile
            profile={"topics":list(profile or []),"history":[],"negative_history":[]}
            self.data["users"][user]=profile
            return profile

        def _negative_feedback_features(self,user,article):
            """Project recent skips into one exclusive symbolic candidate fact.

            Exact, subcategory, and topic matches are deliberately mutually
            exclusive.  They are correlated consequences of one observation and
            must not be revised by PeTTa as three independent pieces of evidence.
            """
            profile=self.data["users"].get(user,{})
            negative_history=(profile.get("negative_history",[])
                              if isinstance(profile,dict) else [])
            negative_history=[str(aid) for aid in negative_history
                              if str(aid) in self._articles]
            aid=str(article["id"])
            if aid in negative_history:
                match="exact"
            else:
                recent=[self._articles[item]
                        for item in negative_history[-LIVE_NEGATIVE_GENERALIZATION_WINDOW:]]
                subcategory=str(article.get("subcategory","unknown"))
                topic=str(article.get("topic",article.get("category","unknown")))
                if (subcategory!="unknown" and any(
                        str(item.get("subcategory","unknown"))==subcategory
                        for item in recent)):
                    match="subcategory"
                elif (topic!="unknown" and any(
                        str(item.get("topic",item.get("category","unknown")))==topic
                        for item in recent)):
                    match="topic"
                else:
                    match="none"
            return {LIVE_NEGATIVE_FEATURE:match}

        def features(self,user,article,history_workspace=None):
            topic=article.get("topic",article.get("category","unknown"))
            article_format=article.get("format",article.get("subcategory","article"))
            profile=self.data["users"].get(user,{})
            history=(profile.get("history",[]) if isinstance(profile,dict) else [])
            if isinstance(profile,dict) and "history" in profile:
                attrs=history_feature_context(
                    article,history,self._articles,
                    entity_vectors=self._article_entity_vectors,
                    text_semantic_vectors=self._article_text_vectors,
                    title_idf_model=self._title_idf_model,
                    transition_model=self.data.get("subcategory_transition_model"),
                    workspace=history_workspace,
                )
                if self._lexical_idf_model:
                    attrs.update(build_lexical_workspace_facts(
                        article, (self._articles.get(aid,{}) for aid in history),
                        self._lexical_idf_model,
                    ))
                if self._semantic_workspace_model:
                    attrs.update(build_semantic_workspace_facts(
                        str(article["id"]),history,self._article_text_vectors,
                        self._semantic_workspace_model,
                    ))
                if self._recency_workspace:
                    attrs.update(build_recency_workspace_facts(
                        str(article["id"]),history,self._article_text_vectors,
                    ))
                if self._llm_workspace:
                    attrs.update(build_llm_workspace_facts(
                        str(article["id"]),history,self._llm_article_annotations,
                    ))
                attrs.update(self._negative_feedback_features(user,article))
                return attrs
            interested=topic in self.user_topics(user)
            recent_subcategories=(profile.get("recent_subcategories",[])
                                  if isinstance(profile,dict) else [])
            transition=subcategory_transition_score(
                article.get("subcategory"),recent_subcategories,
                self.data.get("subcategory_transition_model"),
            )
            return {"topic":topic,"subcategory":article.get("subcategory","unknown"),
                    "format":article_format,"affinity":"high" if interested else "low",
                    "topic_affinity":1.0 if interested else 0.0,
                    "recent_topic_affinity":1.0 if interested else 0.0,
                    "subcategory_affinity_score":0.0,
                    "affinity_level":"high" if interested else "none",
                    "recent_affinity":"high" if interested else "none",
                    "long_affinity":"high" if interested else "none",
                    "history_size_bucket":"unknown","entity_overlap":"unknown",
                    "entity_overlap_detail":"unknown","history_topic_count_bucket":"unknown",
                    "recent_topic_count_bucket":"unknown","topic_rank_bucket":"unknown",
                    "subcategory_affinity":"unknown",
                    "time_bucket":"unknown","ctr_bucket":article.get("ctr_bucket","unknown"),
                    "freshness_bucket":article.get("freshness_bucket","unknown"),
                    "position_bucket":article.get("position_bucket","unknown"),
                    "title_overlap_detail":article.get("title_overlap_detail","unknown"),
                    "entity_long_mean_similarity":None,
                    "title_history_idf_jaccard":None,
                    "recent_subcategory_transition_score":transition,
                    **self._negative_feedback_features(user,article)}

        def contextual_features(self,user,article,context=None,
                                history_workspace=None):
            # A persisted pre-impression snapshot is authoritative, including
            # missing values. Never fill its absent evidence from a later live
            # profile: that can import future history into cold-start replay.
            if context and ("history_size_bucket" in context
                            or any(key in context for key in (*RECENCY_WORKSPACE_FEATURES,*LLM_WORKSPACE_FEATURES))):
                attrs={"topic":article.get("topic",article.get("category","unknown")),
                       "subcategory":article.get("subcategory","unknown"),
                       "format":article.get("format","article")}
            else:
                attrs=self.features(
                    user,article,history_workspace=history_workspace
                )
            if context:
                attrs.update({
                    key:str(context[key])[:200] for key in CONTEXT_FEATURES
                    if key in context and context[key] not in (None,"","unknown")
                    and (not key.startswith("rel_")
                         or self.config.get("relational_evidence_mode")=="chained")
                })
            return {key:str(value) for key,value in attrs.items()
                    if key in CONTEXT_FEATURES and value not in (None,"","unknown")}

        def event_features(self,event):
            # Dataset adapters persist the exact pre-outcome context on each replay
            # event.  Recomputing it from the user's latest profile is both
            # temporally wrong and, for a large corpus, needlessly expensive.
            # Minimal fixtures do not carry such a snapshot and use the live path.
            if any(key in event for key in (
                *RECENCY_WORKSPACE_FEATURES,
                *LLM_WORKSPACE_FEATURES,
                "recent_affinity", "long_affinity", "history_size_bucket",
                "history_topic_count_bucket", "recent_topic_count_bucket",
            )):
                article=self.article(event["article"])
                attrs={
                    "topic":article.get("topic",article.get("category","unknown")),
                    "subcategory":article.get("subcategory","unknown"),
                    "format":article.get("format",article.get("subcategory","article")),
                }
                attrs.update({
                    key:str(event[key])[:200] for key in CONTEXT_FEATURES
                    if key in event and event[key] not in (None,"","unknown")
                    and (not key.startswith("rel_")
                         or self.config.get("relational_evidence_mode")=="chained")
                })
                return {key:str(value) for key,value in attrs.items()
                        if key in CONTEXT_FEATURES and value not in (None,"","unknown")}
            return self.contextual_features(
                event["user"],self.article(event["article"]),event
            )

        def _bounded_features(self,attrs):
            if not self._feature_vocabulary: return attrs
            bounded={}
            for predicate,value in attrs.items():
                allowed=self._feature_vocabulary.get(predicate)
                if not allowed: continue
                bounded[predicate]=value if value in allowed else "other"
            return bounded

        @staticmethod
        def _positive_label(value):
            return value is True or value == 1 or str(value).lower() in POSITIVE|{"1","true","positive"}

        def _case_candidates(self,case):
            raw=case.get("candidates",case.get("candidate_ids",case.get("articles",[])))
            labels=case.get("labels")
            label_map={str(key):value for key,value in labels.items()} if isinstance(labels,dict) else {}
            ids=[]; embedded_relevant=[]
            for index,item in enumerate(raw):
                label=None
                if isinstance(item,dict):
                    aid=item.get("article",item.get("article_id",item.get("news_id",item.get("item",item.get("id")))))
                    label=item.get("label",item.get("clicked",item.get("relevant",item.get("action"))))
                elif isinstance(item,(list,tuple)) and len(item)==2:
                    aid,label=item
                else:
                    aid=item
                    # Accept raw MIND impression tokens such as N12345-1, but only
                    # strip the suffix when the stripped id exists in this dataset.
                    match=re.fullmatch(r"(.+)-([01])",str(aid))
                    if match and str(aid) not in self._articles and match.group(1) in self._articles:
                        aid,label=match.group(1),match.group(2)
                if aid is None: continue
                aid=str(aid)
                if aid not in self._articles: raise ValueError(f"unknown candidate article: {aid}")
                if isinstance(labels,list) and index<len(labels): label=labels[index]
                elif aid in label_map: label=label_map[aid]
                if self._positive_label(label): embedded_relevant.append(aid)
                if aid not in ids: ids.append(aid)
            relevant=case.get("relevant",case.get("clicked_articles",case.get("clicked",case.get("positive_ids",case.get("positive",[])))))
            if isinstance(relevant,(str,int)): relevant=[relevant]
            relevant_ids=[]
            for item in relevant or []:
                if isinstance(item,dict): item=item.get("article",item.get("article_id",item.get("id")))
                if item is not None and str(item) in ids: relevant_ids.append(str(item))
            return ids,list(dict.fromkeys([*relevant_ids,*embedded_relevant]))

        def evaluation_cases(self):
            source=self.data.get("eval_impressions",self.data.get("evaluation",self.data.get("tests",self.data.get("impressions",[]))))
            cases=[]
            for raw in source:
                user=raw.get("user",raw.get("user_id"))
                if user not in self.data["users"]: raise ValueError(f"unknown evaluation user: {user}")
                candidates,relevant=self._case_candidates(raw)
                if candidates and relevant:
                    candidate_context=raw.get("candidate_context",{})
                    supplied_digest=raw.get("training_confirmation_slate_digest")
                    if supplied_digest is not None:
                        from ..evaluation.training_gate import confirmation_slate_digest
                        computed_digest=confirmation_slate_digest(
                            user,candidates,relevant,candidate_context
                        )
                        if supplied_digest!=computed_digest:
                            raise ValueError(
                                "training confirmation slate changed before evaluation"
                            )
                    cases.append({"id":raw.get("id",raw.get("impression_id")),"user":user,
                                  "source_impression_id":raw.get("source_impression_id"),
                                  "candidates":candidates,"relevant":relevant,
                                  "candidate_context":candidate_context,
                                  "training_confirmation_slate_digest":supplied_digest})
            return cases

        def default_candidates(self,user):
            configured=self.data.get("candidate_sets")
            if isinstance(configured,dict) and user in configured:
                candidates,_=self._case_candidates({"candidates":configured[user]})
                if candidates: return candidates
            if isinstance(configured,list):
                for case in configured:
                    if case.get("user",case.get("user_id"))==user:
                        candidates,_=self._case_candidates(case)
                        if candidates: return candidates
            for case in self.evaluation_cases():
                if case["user"]==user: return case["candidates"]
            return list(self._articles)

        def default_candidate_context(self,user):
            source=self.data.get("eval_impressions",self.data.get("evaluation",self.data.get("tests",self.data.get("impressions",[]))))
            for case in source:
                if case.get("user",case.get("user_id"))==user:
                    return case.get("candidate_context",{})
            return {}

        def _feed_pool(self,user):
            """Return unseen corpus IDs; live context is recomputed during scoring."""
            configured=self.data.get("candidate_sets")
            if isinstance(configured,dict) and user in configured:
                source=[{"user":user,"candidates":configured[user]}]
            elif isinstance(configured,list):
                source=[case for case in configured if case.get("user",case.get("user_id"))==user]
            else:
                source=self.data.get("eval_impressions",self.data.get("evaluation",self.data.get("tests",self.data.get("impressions",[]))))
                source=[case for case in source if case.get("user",case.get("user_id"))==user]
            profile=self.data.get("users",{}).get(user,{})
            profile_history=(profile.get("history",[]) if isinstance(profile,dict) else [])
            previously_seen={
                str(aid) for aid in profile_history if str(aid) in self._articles
            }
            for case in source:
                previously_seen.update(
                    str(aid) for aid in case.get("history",[])
                    if str(aid) in self._articles
                )
            pool=[]; seen=set()
            for case in source:
                candidates,_relevant=self._case_candidates(case)
                for aid in candidates:
                    if aid in seen or aid in previously_seen: continue
                    # Evaluation candidate_context belongs to the historical
                    # impression replay. A live feed must derive context from the
                    # current user/article state; carrying every historical
                    # per-item context here explodes proof identities and makes a
                    # cold first page wait on dozens of redundant queries.
                    seen.add(aid); pool.append((aid,{}))
            # MIND usually exposes only one labeled impression per evaluation user.
            # Add the rest of the loaded news corpus so scrolling is a genuine feed;
            # the live scorer derives context afresh for every article.
            remaining=[aid for aid in self._articles if aid not in seen and aid not in previously_seen]
            stable=f'{self.config["random_seed"]}:{user}'.encode()
            local=random.Random(int.from_bytes(hashlib.sha256(stable).digest()[:8],"big"))
            local.shuffle(remaining)
            for aid in remaining:
                seen.add(aid); pool.append((aid,{}))
            return pool

        def _prune_feed_sessions(self,now):
            expired=[session for session,value in self._feed_sessions.items()
                     if now-value["last_access"]>3600]
            for session in expired: self._feed_sessions.pop(session,None)
            if len(self._feed_sessions)>128:
                oldest=sorted(self._feed_sessions,key=lambda key:self._feed_sessions[key]["last_access"])
                for session in oldest[:len(self._feed_sessions)-128]: self._feed_sessions.pop(session,None)

        def _feed_session_for_user(self,session,user):
            state=self._feed_sessions.get(session) if session else None
            if state is not None and state.get("user")!=user:
                raise ValueError("feed session belongs to another user")
            return state

        @staticmethod
        def _acknowledge_feed_deliveries(state,position):
            """Forget pages that a subsequent cursor proves were accepted."""
            state["deliveries"]=[
                delivery for delivery in state.get("deliveries",[])
                if int(delivery["end_position"])>int(position)
            ]

        @staticmethod
        def _record_feed_delivery(state,*,start_position,end_position,rows):
            """Retain a bounded tail so an aborted HTTP response can be undone."""
            if not rows:
                return
            deliveries=state.setdefault("deliveries",[])
            deliveries.append({
                "start_position":int(start_position),
                "end_position":int(end_position),
                "queue_revision":int(state["queue_revision"]),
                "rows":list(rows),
            })
            retained_rows=sum(len(delivery["rows"]) for delivery in deliveries)
            while len(deliveries)>1 and (
                    len(deliveries)>FEED_DELIVERY_HISTORY_LIMIT
                    or retained_rows>FEED_DELIVERY_ROW_LIMIT):
                retained_rows-=len(deliveries[0]["rows"])
                del deliveries[0]

        @staticmethod
        def _recover_unaccepted_feed_deliveries(
                state,*,accepted_position,accepted_revision):
            """Put server-delivered but client-unaccepted rows back in the queue.

            The browser accepts responses atomically and reports the last accepted
            position with feedback.  If an already-running GET crossed the server
            boundary after that position, its rows are recoverable from the
            bounded delivery tail and must participate in the feedback rerank.
            """
            if type(accepted_position) is not int or accepted_position<0:
                raise ValueError("feed_position must be a non-negative integer")
            if type(accepted_revision) is not int or accepted_revision<0:
                raise ValueError("queue_revision must be a non-negative integer")
            if accepted_revision!=state["queue_revision"]:
                raise ValueError("stale feedback queue revision")
            current_position=int(state["position"])
            if accepted_position>current_position:
                raise ValueError("feed_position is ahead of the feed session")

            deliveries=list(state.get("deliveries",[]))
            pending=[delivery for delivery in deliveries
                     if int(delivery["end_position"])>accepted_position]
            recovered=[]
            cursor=accepted_position
            for delivery in pending:
                start=int(delivery["start_position"])
                end=int(delivery["end_position"])
                if start!=cursor or int(delivery["queue_revision"])!=accepted_revision:
                    raise ValueError("feed_position is outside retained delivery history")
                recovered.extend(delivery["rows"])
                cursor=end
            if cursor!=current_position:
                raise ValueError("feed_position is outside retained delivery history")

            if recovered:
                state["queue"]=recovered+state["queue"]
                for row in recovered:
                    key=(str(row["impression"]),str(row["article"]["id"]))
                    state.get("feedback_contexts",{}).pop(key,None)
                    state.get("feedback_positions",{}).pop(key,None)
                state["position"]=accepted_position
            # The accepted prefix belongs to the old queue revision and cannot be
            # useful after feedback creates a new revision.
            state["deliveries"]=[]
            return {
                "accepted_position":accepted_position,
                "server_position_before_recovery":current_position,
                "recovered_rows":len(recovered),
            }

        @staticmethod
        def _live_feedback_signature(row):
            """Return stable, proof-backed online-feedback evidence for one row."""
            evidence=row.get("feedback_evidence",{}) or {}
            declared_ids={str(value) for value in evidence.get("rule_ids",[])}
            proof_rows=[]; proven_ids=set()
            for raw_proof in row.get("proofs",[]) or []:
                proof=str(raw_proof)
                ids=tuple(sorted(
                    LIVE_NEGATIVE_RULE_IDS.intersection(
                        re.findall(r"\bfeedback_skip_[a-z]+\b",proof)
                    )
                ))
                if not ids:
                    continue
                proven_ids.update(ids)
                proof_rows.append((
                    ids,
                    tuple(round(value,12) for value in proof_tv(proof)),
                    re.sub(r"\s+"," ",proof).strip(),
                ))
            # Metadata alone is not causal provenance. Keep only rule IDs actually
            # present in a returned PeTTa proof while retaining mismatches in the
            # diagnostic signature so they cannot compare equal silently.
            rule_ids=tuple(sorted(declared_ids & proven_ids))
            proof_rows=tuple(sorted(set(proof_rows)))
            proof_stv=evidence.get("proof_stv") or {}
            payload={
                "match":str(evidence.get("match","none")),
                "rule_ids":rule_ids,
                "declared_rule_ids":tuple(sorted(declared_ids)),
                "proven_rule_ids":tuple(sorted(proven_ids)),
                "proof_stv":(
                    round(float(proof_stv.get("strength",0.0)),12),
                    round(float(proof_stv.get("confidence",0.0)),12),
                ) if proof_stv else None,
                "proofs":proof_rows,
            }
            encoded=json.dumps(payload,sort_keys=True,separators=(",",":"))
            return {
                **payload,
                "proof_texts":[item[2] for item in proof_rows],
                "signature":hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]
                    if rule_ids and proof_rows else None,
            }

        def feed_page(self,user,*,cursor=None,session=None,limit=None):
            """Return the next page from a window-proof-ranked feed session."""
            with self.lock:
                if user not in self.data["users"]: raise ValueError(f"unknown user: {user}")
                page_size=self.config["top_k"] if limit is None else int(limit)
                if page_size<1 or page_size>1000: raise ValueError("feed limit must be between 1 and 1000")
                cursor_position=None; cursor_revision=None
                if cursor:
                    try:
                        cursor_session,raw_position,raw_revision=cursor.rsplit(":",2)
                        cursor_position=int(raw_position); cursor_revision=int(raw_revision)
                    except (AttributeError,ValueError) as exc: raise ValueError("invalid feed cursor") from exc
                    if session and session!=cursor_session: raise ValueError("feed cursor does not match session")
                    session=cursor_session
                now=time.time(); self._prune_feed_sessions(now); reset=False
                # Validate ownership before applying stale-cursor reset semantics;
                # a token must never become usable merely because its rule version
                # is old.
                state=self._feed_session_for_user(session,user)
                if ((state is not None and state.get("version")!=self.version)
                        or (cursor and state is None)):
                    # A mining/configuration change invalidates the old ordering.
                    # Start a fresh proof-ranked stream and tell the UI to reset.
                    # Validate every explicit session, not only cursor requests: the
                    # HTTP API also accepts ``?session=...`` and must never drain a
                    # queue that was proof-ranked by an older rule snapshot.
                    session=None; cursor_position=None; cursor_revision=None; state=None; reset=True
                if session is None:
                    session=uuid.uuid4().hex
                    pool=self._feed_pool(user)
                    self._feed_sessions[session]={"user":user,"items":pool,"source_position":0,
                                                  "queue":[],"position":0,"last_access":now,
                                                  "version":self.version,"feedback_contexts":{},
                                                  "feedback_positions":{},"deliveries":[],
                                                  "pair_replay":None,
                                                  "queue_revision":0,
                                                  "last_feedback_revision":None}
                    state=self._feed_sessions[session]
                if state is None: raise ValueError("unknown or expired feed session")
                if cursor_position is not None and cursor_position!=state["position"]:
                    raise ValueError("stale feed cursor")
                if cursor_revision is not None and cursor_revision!=state["queue_revision"]:
                    raise ValueError("stale feed cursor revision")
                if cursor_position is not None:
                    # Possession of this cursor proves the preceding response was
                    # accepted by the client, so its rollback copy can be dropped.
                    self._acknowledge_feed_deliveries(state,cursor_position)
                # Treat a bounded slice of the corpus as newly arrived inventory.
                # PeTTaChainer proof-ranks each arrival window before any item from
                # that window can reach the client, keeping first-page latency low.
                window_size=max(page_size,int(self.config["feed_window"]))
                while len(state["queue"])<page_size and state["source_position"]<len(state["items"]):
                    source_start=state["source_position"]
                    source_end=min(len(state["items"]),source_start+window_size)
                    arriving=state["items"][source_start:source_end]
                    candidates=[aid for aid,_context in arriving]
                    contexts={aid:context for aid,context in arriving}
                    rows=(self.score(
                        user,candidates,contexts,limit=0,include_context=True,
                        cache_result=False,
                    ) if candidates else [])
                    state["pair_replay"]=getattr(self,"_last_pair_replay",None)
                    impression=f"live_{session}_{source_start}_{source_end}"
                    queued=[]
                    for row in rows:
                        aid=row["article"]["id"]
                        prepared_context=row.get("_prepared_context")
                        served_context=(dict(prepared_context)
                                        if prepared_context is not None else
                                        self.contextual_features(
                                            user,row["article"],contexts.get(aid)
                                        ))
                        relation=row.get("relational_evidence",{})
                        served_context.update(relation.get("scopes",{}))
                        served_context.update({
                            key:list(value) for key,value
                            in relation.get("proof_ids",{}).items()
                        })
                        public_row={key:value for key,value in row.items()
                                    if key!="_prepared_context"}
                        queued.append({**public_row,"context":served_context,
                                       "impression":impression,
                                       "queue_revision":state["queue_revision"]})
                    state["queue"].extend(queued)
                    state["source_position"]=source_end
                delivery_start=state["position"]
                rows=state["queue"][:page_size]; del state["queue"][:len(rows)]
                delivery_end=delivery_start+len(rows)
                # Only rows that actually crossed the HTTP boundary are eligible
                # for feedback.  Previously all 40 proof-ranked rows were
                # registered before five were emitted, allowing a caller to react
                # to an item the browser had never received.
                for row in rows:
                    aid=str(row["article"]["id"])
                    key=(str(row["impression"]),aid)
                    state["feedback_contexts"][key]=row["context"]
                    state["feedback_positions"][key]=delivery_end
                state["position"]=delivery_end; state["last_access"]=now
                self._record_feed_delivery(
                    state,start_position=delivery_start,
                    end_position=delivery_end,rows=rows,
                )
                has_more=bool(state["queue"] or state["source_position"]<len(state["items"]))
                next_cursor=(f"{session}:{state['position']}:{state['queue_revision']}"
                             if has_more else None)
                return {"feed":rows,"session":session,"cursor":next_cursor,
                        "next_cursor":next_cursor,"has_more":has_more,
                        "position":state["position"],
                        "remaining":len(state["items"])-state["position"],"total_candidates":len(state["items"]),
                        "rule_version":self.version,"reset":reset,
                        "queue_revision":state["queue_revision"],
                        "last_feedback_revision":state["last_feedback_revision"]}

        def _rerank_unserved_queue(self,session_id,state,*,action,article):
            """Proof-rank one live session's already-arrived, unserved inventory."""
            before_rows=list(state["queue"])
            before=[str(row["article"]["id"]) for row in before_rows]
            before_by_article={
                str(row["article"]["id"]):(index+1,row)
                for index,row in enumerate(before_rows)
            }
            state["queue_revision"]+=1
            revision=state["queue_revision"]
            rerank_mode="full"
            recomputed_candidates=len(before)
            if before:
                can_reuse_pairwise=(
                    action=="skip"
                    and all(isinstance(row.get("pairwise_score"),(int,float))
                            for row in before_rows)
                )
                replay_source=state.get("pair_replay")
                replay_available=(self._pairwise_replay_plan(
                    before_rows,state.get("pair_replay")
                ) if can_reuse_pairwise else None)
                can_reuse_pairwise=(can_reuse_pairwise
                                    and replay_available is not None)
                changed=[]
                revised_contexts={}
                if can_reuse_pairwise:
                    for aid in before:
                        row=before_by_article[aid][1]
                        revised_context=dict(row.get("context",{}))
                        old_match=str(
                            revised_context.get(
                                LIVE_NEGATIVE_FEATURE,
                                row.get("feedback_evidence",{}).get("match","none"),
                            )
                        )
                        new_match=self._negative_feedback_features(
                            state["user"],self.article(aid)
                        )[LIVE_NEGATIVE_FEATURE]
                        revised_context[LIVE_NEGATIVE_FEATURE]=new_match
                        revised_contexts[aid]=revised_context
                        if old_match!=new_match:
                            changed.append(aid)
                if can_reuse_pairwise:
                    changed_rows=(self.score(
                        state["user"],changed,
                        contexts={aid:revised_contexts[aid] for aid in changed},
                        limit=0,
                        include_context=True,apply_pairwise=False,
                        cache_result=False,
                    ) if changed else [])
                    changed_by_article={
                        str(row["article"]["id"]):row for row in changed_rows
                    }
                    reranked=[]
                    for aid in before:
                        previous=before_by_article[aid][1]
                        if aid not in changed_by_article:
                            reranked.append(dict(previous))
                            continue
                        updated=changed_by_article[aid]
                        reranked.append(updated)
                    reranked.sort(key=lambda row:(
                        -row["score"],-row["stv"]["strength"],
                        -row["stv"]["confidence"],
                        -row["tie_break"]["topic_prior"],
                        -row["tie_break"]["format_prior"],
                        -row["tie_break"]["subcategory_prior"],
                        row["article"]["id"],
                    ))
                    plan,proof_map=self._pairwise_replay_plan(
                        reranked,replay_source
                    )
                    reranked=self._pairwise_rank(
                        reranked,(),plan=plan,proof_map=proof_map,
                        reasoner_timeout_sec=self._serving_reasoner_timeout(),
                    )
                    rerank_mode="incremental_point_reuse_pairwise"
                    recomputed_candidates=len(changed)
                else:
                    reranked=self.score(
                        state["user"],before,contexts={},limit=0,
                        include_context=True,cache_result=False,
                    )
                impression=(f"live_{session_id}_{state['position']}_"
                            f"{state['source_position']}")
                state["queue"]=[]
                for row in reranked:
                    aid=str(row["article"]["id"])
                    prepared=row.get("_prepared_context")
                    context=(dict(prepared) if prepared is not None else
                             dict(before_by_article[aid][1].get("context",{})))
                    if not context:
                        context=self.contextual_features(
                            state["user"],row["article"],None
                        )
                    relation=row.get("relational_evidence",{})
                    context.update(relation.get("scopes",{}))
                    context.update({
                        key:list(value) for key,value
                        in relation.get("proof_ids",{}).items()
                    })
                    public_row={key:value for key,value in row.items()
                                if key!="_prepared_context"}
                    state["queue"].append({
                        **public_row,"context":context,"impression":impression,
                        "queue_revision":revision,
                    })
            after=[str(row["article"]["id"]) for row in state["queue"]]
            state["version"]=self.version
            old_positions={aid:index+1 for index,aid in enumerate(before)}
            movements=[
                {"article":aid,"from":old_positions[aid],"to":index+1,
                 "delta":old_positions[aid]-(index+1)}
                for index,aid in enumerate(after)
                if old_positions.get(aid)!=index+1
            ]
            affected=[]; observed=[]
            for after_rank,row in enumerate(state["queue"],1):
                aid=str(row["article"]["id"])
                before_rank,before_row=before_by_article[aid]
                before_evidence=self._live_feedback_signature(before_row)
                after_evidence=self._live_feedback_signature(row)
                rule_ids=list(after_evidence["rule_ids"])
                if not rule_ids:
                    continue
                before_score=float(before_row.get("score",0.0))
                after_score=float(row.get("score",0.0))
                before_ranking=float(before_row.get("ranking_score",before_score))
                after_ranking=float(row.get("ranking_score",after_score))
                introduced=before_evidence["signature"] is None
                evidence_changed=(
                    after_evidence["signature"] is not None
                    and after_evidence["signature"]!=before_evidence["signature"]
                )
                score_decreased=after_score<before_score
                ranking_score_decreased=after_ranking<before_ranking
                diagnostic={
                    "article":aid,
                    "before_rank":before_rank,"after_rank":after_rank,
                    "rank_delta":after_rank-before_rank,
                    "before_score":before_score,"after_score":after_score,
                    "score_delta":round(after_score-before_score,8),
                    "before_ranking_score":before_ranking,
                    "after_ranking_score":after_ranking,
                    "ranking_score_delta":round(after_ranking-before_ranking,8),
                    "score_method":row.get("score_method"),
                    "before_match":before_evidence["match"],
                    "match":after_evidence["match"],
                    "before_rule_ids":list(before_evidence["rule_ids"]),
                    "rule_ids":rule_ids,
                    "before_proofs":before_evidence["proof_texts"],
                    "proofs":after_evidence["proof_texts"],
                    "before_feedback_signature":before_evidence["signature"],
                    "after_feedback_signature":after_evidence["signature"],
                    "feedback_evidence_introduced":introduced,
                    "feedback_evidence_changed":evidence_changed,
                    "own_score_decreased":score_decreased,
                    "own_ranking_score_decreased":ranking_score_decreased,
                    "causal_eligible":bool(
                        evidence_changed and (score_decreased or ranking_score_decreased)
                    ),
                }
                observed.append(diagnostic)
                if evidence_changed:
                    affected.append(diagnostic)
            # A rank can worsen merely because some *other* row moved. Causality is
            # certified only when this candidate gained/changed proof-backed
            # feedback evidence and its own point or final ranking score fell.
            causal_demotion=next((
                item for item in affected if item["causal_eligible"]
            ),None)
            has_more=bool(state["queue"] or state["source_position"]<len(state["items"]))
            next_cursor=(f"{session_id}:{state['position']}:{revision}"
                         if has_more else None)
            summary={
                "session":session_id,"revision":revision,"action":action,
                "feedback_article":str(article),"reranked_candidates":len(after),
                "rerank_mode":rerank_mode,
                "recomputed_candidates":recomputed_candidates,
                "reused_candidates":max(0,len(after)-recomputed_candidates),
                "pairwise_reused":rerank_mode=="incremental_point_reuse_pairwise",
                "changed_positions":len(movements),
                "top_before":before[0] if before else None,
                "top_after":after[0] if after else None,
                "movements":movements[:20],"negative_proof_candidates":affected[:20],
                "negative_proof_observations":observed[:20],
                "unchanged_negative_proof_candidates":sum(
                    not item["feedback_evidence_changed"] for item in observed
                ),
                "causal_demotion":causal_demotion,"next_cursor":next_cursor,
                "reasoner":"PeTTaChainer",
            }
            state["last_feedback_revision"]=summary
            state["last_access"]=time.time()
            return summary

        @staticmethod
        def candidate_case(user,aid,attrs):
            signature=json.dumps(attrs,sort_keys=True,separators=(",",":"),ensure_ascii=False)
            # Serving rules only inspect these mined feature predicates. Reusing a
            # grounded context for identical feature triples is semantically exact
            # and keeps a large impression replay bounded.
            digest=hashlib.blake2s(signature.encode("utf-8"),digest_size=10).hexdigest()
            return f"candidate_{digest}"

        @staticmethod
        def _relational_proof_references(context):
            """Return normalized relation-proof references from one context.

            Proof IDs are audit dependencies rather than scoring features. Keep
            them out of fpMiner while still treating malformed or duplicate
            references as invalid provenance at the serving boundary.
            """
            if not isinstance(context,dict):
                return {}
            found={}
            for proof_field in RELATIONAL_PROOF_FIELDS.values():
                if proof_field not in context:
                    continue
                references=context[proof_field]
                if not isinstance(references,(list,tuple)) or any(
                        not isinstance(reference,str) or not reference
                        for reference in references):
                    raise ValueError(
                        "relational proof references must be nonempty string lists"
                    )
                normalized=tuple(references)
                if len(normalized)!=len(set(normalized)):
                    raise ValueError("relational proof references must be unique")
                found[proof_field]=normalized
            return found

        def _validate_relational_context_provenance(self,context,*,user,article):
            """Bind each relational scope to one coherent proved history snapshot."""
            references=self._relational_proof_references(context)
            immutable=self.data.get("relational_proof_ledger",{})
            if not isinstance(immutable,dict):
                immutable={}
            live=getattr(self,"_live_relational_proof_ledger",{})
            expected_families={
                REL_ENTITY_CONTINUITY_SCOPE:"wikidata_entity_continuity",
                REL_CONCEPT_CONTINUITY_SCOPE:"canonical_concept_continuity",
            }
            causal_histories=set()
            for scope_field,proof_field in RELATIONAL_PROOF_FIELDS.items():
                field_references=references.get(proof_field,())
                scope=context.get(scope_field)
                if scope is not None and scope not in {
                        "unknown","none","older","recent"}:
                    raise ValueError("invalid relational scope")
                if scope in {"older","recent"} and not field_references:
                    raise ValueError(
                        "a positive relational scope requires proof references"
                    )
                if field_references and scope not in {"older","recent"}:
                    raise ValueError(
                        "positive relational proof references require a positive scope"
                    )
                records=[]
                for proof_id in field_references:
                    immutable_record=immutable.get(proof_id)
                    live_record=live.get(proof_id)
                    if (immutable_record is not None and live_record is not None
                            and immutable_record!=live_record):
                        raise RuntimeError("relational proof ID collision")
                    record=(live_record if live_record is not None
                            else immutable_record)
                    if not isinstance(record,dict):
                        raise ValueError("relational proof reference is not available")
                    if (str(record.get("user_id"))!=str(user)
                            or str(record.get("candidate_id"))!=str(article)):
                        raise ValueError(
                            "relational proof reference belongs to another context"
                        )
                    if record.get("relation_family")!=expected_families[scope_field]:
                        raise ValueError(
                            "relational proof reference belongs to another family"
                        )
                    records.append(record)
                if records:
                    record_scopes={str(record.get("recency"))
                                   for record in records}
                    derived_scope=("recent" if "recent" in record_scopes
                                   else "older")
                    if not record_scopes.issubset({"older","recent"}) or scope!=derived_scope:
                        raise ValueError(
                            "relational scope disagrees with proof recency"
                        )
                    for identity_field in (
                            "case_id","scope_id","causal_history_id"):
                        identities={record.get(identity_field) for record in records}
                        if len(identities)!=1 or None in identities:
                            raise ValueError(
                                "relational proof references mix incompatible contexts"
                            )
                    causal_histories.add(records[0]["causal_history_id"])
            if len(causal_histories)>1:
                raise ValueError(
                    "relational proof families use different causal histories"
                )
            return references

        def _referenced_live_relational_proof_ids(self):
            """Return proof IDs that must survive the next worker promotion."""
            references=set()
            events=self.data.get("events",[])
            if isinstance(events,list):
                start=max(0,int(getattr(self,"_offline_event_count",0)))
                for event in events[start:]:
                    for values in self._relational_proof_references(event).values():
                        references.update(values)
            for state in getattr(self,"_feed_sessions",{}).values():
                if not isinstance(state,dict):
                    continue
                contexts=state.get("feedback_contexts",{})
                if not isinstance(contexts,dict):
                    continue
                for context in contexts.values():
                    for values in self._relational_proof_references(context).values():
                        references.update(values)
            return references

        def _retained_live_relational_proofs(self,*additional_ledgers):
            """Snapshot live proof records referenced by events or served cards."""
            immutable=self.data.get("relational_proof_ledger",{})
            if not isinstance(immutable,dict):
                immutable={}
            ledgers=[getattr(self,"_live_relational_proof_ledger",{}),
                     *additional_ledgers]
            combined={}
            for ledger in ledgers:
                if not isinstance(ledger,dict):
                    continue
                for proof_id,record in ledger.items():
                    previous=combined.get(proof_id)
                    if previous is not None and previous!=record:
                        raise RuntimeError("relational proof ID collision")
                    combined[proof_id]=record
            retained={}
            missing=[]
            for proof_id in sorted(self._referenced_live_relational_proof_ids()):
                if proof_id in immutable:
                    continue
                record=combined.get(proof_id)
                if record is None:
                    missing.append(proof_id)
                else:
                    retained[proof_id]=copy.deepcopy(record)
            if missing:
                raise RuntimeError(
                    "live relational proof references would become dangling: "
                    +", ".join(missing[:5])
                )
            return retained

        def _live_relational_features(
                self,user,candidates,required_features,*,timeout_sec=None):
            """Derive live candidate relations through the active PeTTa worker.

            Historical replay contexts already contain their causal, pre-outcome
            relational projection. This path is exclusively for a live profile: a
            positive interaction changes its ordered click history, creating new
            versioned relation scopes and therefore new proof queries. Every
            positive conclusion remains gated by a returned two-hop PeTTa proof.
            """
            selected=set(required_features).intersection({
                REL_ENTITY_CONTINUITY_SCOPE,
                REL_CONCEPT_CONTINUITY_SCOPE,
            })
            if (not selected
                    or self.config.get("relational_evidence_mode")!="chained"):
                return {str(aid):{} for aid in candidates}
            timeout_sec=(self._serving_reasoner_timeout()
                         if timeout_sec is None else float(timeout_sec))
            profile=self.data["users"].get(user,{})
            raw_history=(profile.get("history")
                         if isinstance(profile,dict) and "history" in profile
                         else None)
            history_available=(
                isinstance(raw_history,(list,tuple))
                and all(isinstance(aid,str) and bool(aid) for aid in raw_history)
            )
            normalized_history=(tuple(raw_history) if history_available else ())
            history_truncated=len(normalized_history)>RELATIONAL_LIVE_HISTORY_LIMIT
            # Unknown IDs are evidence gaps. Preserve them so the builders emit
            # ``unknown`` instead of manufacturing a closed-world ``none`` from a
            # silently filtered history.
            history=normalized_history[-RELATIONAL_LIVE_HISTORY_LIMIT:]
            entity_observations={}; concept_observations={}
            plans=[]
            for aid in candidates:
                entity_plan,concept_plan=build_relational_plans(
                    str(aid),history,self._articles,
                    self._llm_article_annotations,user_id=str(user),
                    entity_observation_cache=entity_observations,
                    concept_observation_cache=concept_observations,
                )
                if REL_ENTITY_CONTINUITY_SCOPE in selected:
                    plans.append((str(aid),REL_ENTITY_CONTINUITY_SCOPE,
                                  entity_plan,reduce_relational_proofs))
                if REL_CONCEPT_CONTINUITY_SCOPE in selected:
                    plans.append((str(aid),REL_CONCEPT_CONTINUITY_SCOPE,
                                  concept_plan,reduce_concept_relational_proofs))

            missing=[]; statements=set()
            for aid,field,plan,reducer in plans:
                cache_key=(field,plan.case_id,history_truncated,history_available)
                if cache_key in self._relational_feature_cache:
                    continue
                missing.append((aid,field,plan,reducer))
                if plan.requires_query:
                    statements.update(plan.statements)
            cache_limit=int(self.config.get(
                "max_proof_cache_entries",DEFAULT_MAX_PROOF_CACHE_ENTRIES
            ))
            if len(self._relational_feature_cache)+len(missing)>cache_limit:
                raise RuntimeError(
                    "relational feature cache limit exceeded; promote a fresh "
                    "rule snapshot"
                )
            new_statements=sorted(
                statement for statement in statements
                if statement not in self._loaded_relational_statements
            )
            for offset in range(0,len(new_statements),1000):
                statement_batch=new_statements[offset:offset+1000]
                self.engine.add_atoms_no_check(
                    statement_batch,timeout_sec=timeout_sec
                )
                # Publish each acknowledged mutation immediately. If a later batch
                # fails, the worker journal recovers these facts and a retry must
                # not insert named duplicates and inflate revision confidence.
                self._loaded_relational_statements.update(statement_batch)

            proof_results={}
            queryable=[item for item in missing if item[2].requires_query]
            query_roots=[
                (item,root) for item in queryable for root in item[2].proof_roots
            ]
            if any(not item[2].proof_roots for item in queryable):
                raise RuntimeError(
                    "a queryable live relation has no specific proof roots"
                )
            query_batch=max(1,int(self.config.get("query_batch_size",512)))
            for offset in range(0,len(query_roots),query_batch):
                batch=query_roots[offset:offset+query_batch]
                results=self.engine.query_many(
                    [root.query for _item,root in batch],
                    steps=max(8,RELATIONAL_QUERY_STEPS_PER_ROOT*len(batch)),
                    timeout_sec=timeout_sec,
                )
                if not isinstance(results,(list,tuple)) or len(results)!=len(batch):
                    raise RuntimeError(
                        "PeTTaChainer returned an invalid live relational batch"
                    )
                self._relational_query_calls+=1
                self._relational_query_roots+=len(batch)
                for ((_aid,_field,plan,_reducer),root),proofs in zip(batch,results):
                    if not isinstance(proofs,(list,tuple)) or not proofs:
                        raise RuntimeError(
                            "a required live relational proof root returned no "
                            "PeTTa proof: "
                            f"{root.origin_id}/{root.matched_value_id}"
                        )
                    proof_results.setdefault(plan.case_id,[]).extend(proofs)

            staged_cache={}; staged_ledger={}
            for _aid,field,plan,reducer in missing:
                facts,ledger=reducer(plan,proof_results.get(plan.case_id,()))
                facts=dict(facts)
                if ((history_truncated or not history_available)
                        and facts.get(field)=="none"):
                    # No match inside the bounded window says nothing about the
                    # omitted prefix. This is abstention, not negative knowledge.
                    facts[field]="unknown"
                staged_cache[(
                    field,plan.case_id,history_truncated,history_available,
                )]=facts
                for proof_id,record in ledger.items():
                    previous=(self._live_relational_proof_ledger.get(proof_id)
                              or staged_ledger.get(proof_id))
                    if previous is not None and previous!=record:
                        raise RuntimeError("stable live relational proof ID collision")
                    staged_ledger[proof_id]=record
            self._relational_feature_cache.update(staged_cache)
            self._live_relational_proof_ledger.update(staged_ledger)

            projected={str(aid):{} for aid in candidates}
            for aid,field,plan,_reducer in plans:
                facts=self._relational_feature_cache[
                    (field,plan.case_id,history_truncated,history_available)
                ]
                value=facts.get(field)
                proof_field=RELATIONAL_PROOF_FIELDS[field]
                projected[aid][proof_field]=list(facts.get(proof_field,()))
                # Unknown means source annotations were incomplete. It must
                # abstain, not become a provider-coverage ranking signal.
                if value!="unknown":
                    projected[aid][field]=value
            return projected

        def _candidate_specs(self,user,candidates,contexts=None):
            contexts=contexts or {}
            # A proof can only inspect predicates selected by the active mining
            # profile.  Projecting grounded contexts to that profile is both
            # semantically exact and important for a large MIND replay: temporal
            # fields that are not in the rule vocabulary must not create a new
            # candidate identity (and therefore a new PeTTa query) for every
            # impression.
            active=set(FEATURE_PROFILES[self.config["feature_profile"]])
            # A persisted replay value is authoritative even when it is
            # ``unknown``. Derive only fields absent from the supplied historical
            # context; this prevents current live history leaking into replay.
            live_required={
                feature for feature in active
                if feature in {
                    REL_ENTITY_CONTINUITY_SCOPE,
                    REL_CONCEPT_CONTINUITY_SCOPE,
                }
                and any(feature not in (contexts.get(str(aid)) or {})
                        for aid in candidates)
            }
            live_relational=self._live_relational_features(
                user,candidates,live_required,
                timeout_sec=self._serving_reasoner_timeout(),
            )
            profile=self.data["users"].get(user,{})
            history=(profile.get("history",[]) if isinstance(profile,dict) else [])
            needs_live_history=any(
                not (contexts.get(str(aid)) or {}).get("history_size_bucket")
                for aid in candidates
            )
            history_workspace=(prepare_history_feature_workspace(
                history,self._articles,
                entity_vectors=self._article_entity_vectors,
                text_semantic_vectors=self._article_text_vectors,
            ) if isinstance(profile,dict) and "history" in profile
                 and needs_live_history else None)
            specs=[]
            feature_started=time.perf_counter()
            def prepare(aid):
                supplied=contexts.get(aid) or {}
                raw_attrs=self.contextual_features(
                    user,self.article(aid),contexts.get(aid),
                    history_workspace=history_workspace,
                )
                for feature,value in live_relational.get(str(aid),{}).items():
                    if feature not in supplied:
                        raw_attrs[feature]=value
                return aid,supplied,raw_attrs
            executor=getattr(self,"_candidate_feature_executor",None)
            if executor is not None and len(candidates)>1:
                prepared=list(executor.map(prepare,candidates))
            else:
                prepared=[prepare(aid) for aid in candidates]
            feature_seconds=time.perf_counter()-feature_started
            projection_seconds=0.0
            case_serialization_seconds=0.0
            for aid,supplied,raw_attrs in prepared:
                started=time.perf_counter()
                attrs=self._bounded_features(raw_attrs)
                attrs={key:value for key,value in attrs.items() if key in active}
                # Online negative feedback is a serving fact, not a mined feature
                # vocabulary value.  Keep it available under every experimental
                # profile so the generic PeTTa feedback policy cannot silently
                # disappear when an operator changes the mining profile.
                if LIVE_NEGATIVE_FEATURE in raw_attrs:
                    attrs[LIVE_NEGATIVE_FEATURE]=raw_attrs[LIVE_NEGATIVE_FEATURE]
                projection_seconds+=time.perf_counter()-started
                started=time.perf_counter()
                relational_refs={}
                for _scope_field,proof_field in RELATIONAL_PROOF_FIELDS.items():
                    references=raw_attrs.get(
                        proof_field,supplied.get(proof_field,())
                    )
                    if isinstance(references,(list,tuple)):
                        relational_refs[proof_field]=tuple(map(str,references))
                case_identity=dict(attrs)
                nonempty_refs={key:value for key,value in relational_refs.items()
                               if value}
                if nonempty_refs:
                    # A proof-positive categorical value may be reusable, but its
                    # source occurrence is not interchangeable. Bind the scoring
                    # case to exact proof IDs so returned provenance cannot drift
                    # across users or historical snapshots with the same scope.
                    case_identity["__relational_proof_ids__"]=nonempty_refs
                case=self.candidate_case(user,aid,case_identity)
                previous=self._candidate_relational_proof_refs.get(case)
                if previous is not None and previous!=relational_refs:
                    raise RuntimeError("candidate relation-provenance collision")
                self._candidate_relational_proof_refs[case]=relational_refs
                case_serialization_seconds+=time.perf_counter()-started
                specs.append((aid,case,attrs,raw_attrs))
            self._last_candidate_preparation_profile={
                "candidate_history_feature_calculation_seconds":feature_seconds,
                "candidate_feature_projection_seconds":projection_seconds,
                "candidate_case_serialization_seconds":case_serialization_seconds,
                "candidates":len(specs),
            }
            return specs


    return ServingMixin
