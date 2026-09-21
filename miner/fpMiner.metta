(= (candidatePatternMaker $dbspace ($link left $rValue) $minsup)
    (if (sup-eval $dbspace ($link $x $rValue) $minsup)
        ($link "$x" $rValue)
        (empty))
)
( = (absSpeCan $dbspace $minsup)
    (let $special (alpha-unique-atom
                      (collapse
                          (match $dbspace ($link $x $y)
                              ($link left $y))))
        (candidatePatternMaker $dbspace (superpose $special) $minsup))
)
(= (formatter
      (conjunct $conjunct) $dbspace $minSup $target-value $evidence-k)
   (if (recommendation-pattern-has-target? $conjunct $target-value)
       (let $support (counter $dbspace $conjunct)
         (if (>= $support $minSup)
             (let* (($CTV (recommendation-ctv-with-support
                              $conjunct $dbspace $support
                              $target-value $evidence-k))
                    ($organized (promote-engagement-conj $conjunct)))
               (supportOf ($organized $CTV) $support))
             (empty)))
       (empty)))

(= (frequency-pattern-miner
      $dbspace $minsup $depth $target-value $evidence-k)
   (let $candidatePatterns (collapse (absSpeCan $dbspace $minsup))
     (let $conjuncts (unique-combinations-star $candidatePatterns $depth)
       (collapse
         (formatter (superpose $conjuncts) $dbspace $minsup
                    $target-value $evidence-k)))))

;; Recommendation-local compatibility entry point.  This miner's closed-case
;; contract has always used click as the positive target; generic callers must
;; use the explicit form instead of assuming another engagement label.
(= (frequency-pattern-miner $dbspace $minsup $depth)
   (frequency-pattern-miner $dbspace $minsup $depth "click" 800.0))

;; Retain a candidate premise only when its unary binary-predictor AUC is at
;; least the requested gain above random.  This is an Apriori-style target
;; gate: deeper conjunctions are assembled only from premises which already
;; carry positive target information.  Counting, AUC, WRAcc, information gain,
;; log odds and CTV construction all remain inside PeTTa.
(= (target-aware-candidate
      $candidate $dbspace $minsup $target-value $evidence-k $min-auc-gain)
   (if (recommendation-engagement? $candidate)
       (if (recommendation-pattern-has-target? $candidate $target-value)
           $candidate
           (empty))
       (let $unary-pattern
            (promote-placeholders
              (, $candidate (engagement "$x" $target-value)) ("$x"))
         (let $support (counter $dbspace $unary-pattern)
           (if (>= $support $minsup)
               (let (TargetMetrics $_ctv $_counts $_auc
                      (AUC-Gain $gain) $_youden $_wracc $_ig $_log-odds)
                    (recommendation-target-metrics-with-support
                      $unary-pattern $dbspace $support
                      $target-value $evidence-k)
                 (if (>= $gain $min-auc-gain) $candidate (empty)))
               (empty))))))

(= (target-aware-formatter
      (conjunct $conjunct) $dbspace $minsup $target-value
      $evidence-k $min-auc-gain)
   (if (recommendation-pattern-has-target? $conjunct $target-value)
       (let $support (counter $dbspace $conjunct)
         (if (>= $support $minsup)
             (let (TargetMetrics $ctv $counts (AUC $auc)
                    (AUC-Gain $gain) (Youden-J $youden)
                    (WRAcc $wracc) (Information-Gain $information-gain)
                    (Log-Odds $log-odds))
                  (recommendation-target-metrics-with-support
                    $conjunct $dbspace $support $target-value $evidence-k)
               (if (>= $gain $min-auc-gain)
                   (let $organized (promote-engagement-conj $conjunct)
                     (targetScoreOf
                       ($organized $ctv) $counts
                       (AUC $auc) (AUC-Gain $gain) (Youden-J $youden)
                       (WRAcc $wracc) (Information-Gain $information-gain)
                       (Log-Odds $log-odds) $support))
                   (empty)))
             (empty)))
       (empty)))

(= (target-aware-frequency-pattern-miner
      $dbspace $minsup $depth $target-value $evidence-k $min-auc-gain)
   (let $candidate-patterns (collapse (absSpeCan $dbspace $minsup))
     (let $target-candidates
          (collapse
            (target-aware-candidate
              (superpose $candidate-patterns) $dbspace $minsup
              $target-value $evidence-k $min-auc-gain))
       (let $conjuncts
            (unique-combinations-star $target-candidates $depth)
         (collapse
           (target-aware-formatter
             (superpose $conjuncts) $dbspace $minsup $target-value
             $evidence-k $min-auc-gain))))))

(= (target-candidate-in-family $candidate $functors)
   (if (is-member (functor-of $candidate) $functors)
       $candidate
       (empty)))

;; Conditional expansion reuses the unary statistics already needed by the
;; target gate.  Carrying precision with the candidate avoids recounting the
;; same parent once for every child conjunction.
(= (target-aware-candidate-with-precision
      $candidate $dbspace $minsup $target-value $evidence-k $min-auc-gain)
   (let $unary-pattern
        (promote-placeholders
          (, $candidate (engagement "$x" $target-value)) ("$x"))
     (let $support (counter $dbspace $unary-pattern)
       (if (>= $support $minsup)
           (let (TargetMetrics
                  (CTV (STV $precision $_positive-confidence) $_negative-tv)
                  $_counts $_auc (AUC-Gain $gain) $_youden $_wracc $_ig
                  $_log-odds)
                (recommendation-target-metrics-with-support
                  $unary-pattern $dbspace $support $target-value $evidence-k)
             (if (>= $gain $min-auc-gain)
                 (TargetCandidate $candidate $precision)
                 (empty)))
           (empty)))))

;; A conjunction must improve its target precision over both immediate unary
;; parents.  This prevents a context clause from receiving credit merely for
;; restating evidence already carried by the semantic clause.  All counts and
;; arithmetic stay inside PeTTa; the host only decodes the emitted atom.
(= (conditional-target-aware-formatter
      (conjunct $conjunct) $parent-precision $dbspace $minsup
      $target-value $evidence-k $min-auc-gain $min-incremental-precision)
   (let $support (counter $dbspace $conjunct)
     (if (>= $support $minsup)
         (let (TargetMetrics
                (CTV (STV $precision $positive-confidence) $negative-tv)
                (Contingency $n11 $n10 $n01 $n00)
                (AUC $auc) (AUC-Gain $gain) (Youden-J $youden)
                (WRAcc $wracc) (Information-Gain $information-gain)
                (Log-Odds $log-odds))
              (recommendation-target-metrics-with-support
                $conjunct $dbspace $support $target-value $evidence-k)
           (let* (($incremental-precision (- $precision $parent-precision))
                  ($antecedent-support (+ $n11 $n10))
                  ($population (+ $n11 (+ $n10 (+ $n01 $n00))))
                  ($incremental-wracc
                     (* (// $antecedent-support $population)
                        $incremental-precision)))
             (if (and (>= $gain $min-auc-gain)
                      (> $incremental-precision
                         $min-incremental-precision))
                 (let $organized (promote-engagement-conj $conjunct)
                   (targetScoreOf
                     ($organized
                       (CTV
                         (STV $precision $positive-confidence)
                         $negative-tv))
                     (Contingency $n11 $n10 $n01 $n00)
                     (AUC $auc) (AUC-Gain $gain)
                     (Youden-J $youden) (WRAcc $wracc)
                     (Information-Gain $information-gain)
                     (Log-Odds $log-odds)
                     (Parent-Precision $parent-precision)
                     (Incremental-Precision $incremental-precision)
                     (Incremental-WRAcc $incremental-wracc)
                     $support))
                 (empty))))
         (empty))))

;; Specialized two-premise expansion for recommendation: one directional
;; semantic signal and one orientation-invariant context gate.  Partitioning
;; the pools before the Cartesian expansion avoids generating semantic x
;; semantic and context x context candidates which the architecture would
;; reject later anyway.
(= (conditional-target-aware-frequency-pattern-miner
      $dbspace $minsup $semantic-functors $context-functors
      $target-value $evidence-k $min-auc-gain $min-incremental-precision)
   (let $candidate-patterns (collapse (absSpeCan $dbspace $minsup))
     (let* (($semantic-pool
               (collapse
                 (target-candidate-in-family
                   (superpose $candidate-patterns) $semantic-functors)))
            ($context-pool
               (collapse
                 (target-candidate-in-family
                   (superpose $candidate-patterns) $context-functors)))
            ($semantic-candidates
               (collapse
                 (target-aware-candidate-with-precision
                   (superpose $semantic-pool) $dbspace $minsup
                   $target-value $evidence-k $min-auc-gain)))
            ($context-candidates
               (collapse
                 (target-aware-candidate-with-precision
                   (superpose $context-pool) $dbspace $minsup
                   $target-value $evidence-k $min-auc-gain))))
       (let* (((TargetCandidate $semantic $semantic-precision)
                 (superpose $semantic-candidates))
              ((TargetCandidate $context $context-precision)
                 (superpose $context-candidates))
              ($parent-precision
                 (max $semantic-precision $context-precision))
              ($conjunction
                (promote-placeholders
                  (, $semantic $context (engagement "$x" $target-value))
                  ("$x"))))
         (collapse
           (conditional-target-aware-formatter
             (conjunct $conjunction) $parent-precision $dbspace $minsup
             $target-value $evidence-k $min-auc-gain
             $min-incremental-precision))))))
