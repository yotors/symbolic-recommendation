!(import! &self ../helpers)
!(import! &self ../fpMiner)

; Test dataset: simple, deterministic supports
!(add-atom &space (topic 0 AI))
!(add-atom &space (engagement 0 low))

!(add-atom &space (topic 1 AI))
!(add-atom &space (engagement 1 low))

!(add-atom &space (topic 3 Gardening))
!(add-atom &space (engagement 3 low))

; With minsup=2 and depth=2, only (topic $V0 "AI") and (length $V0 "low") qualify.
; Their conjunction has support 2 (articles 0 and 1). The formatter sorts atoms lexicographically.
!(frequency-pattern-miner &space 1 2)