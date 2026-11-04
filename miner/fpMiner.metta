!(import! &self ./conjunctionExpansion)

;; Target-aware rendering for relational conjunctions. Output stays
;; fpMiner-compatible while graph-path multiplicity is deduplicated by case.
(= (target-aware-formatter
      $pattern $dbspace $min-support $target-value $evidence-k $length-cost)
   (if (recommendation-pattern-has-target? $pattern $target-value)
       (let $support
            (recommendation-pattern-support $dbspace $pattern $target-value)
         (if (and (>= $support $min-support)
                  (recommendation-mdl-accepts?
                    $pattern $dbspace $target-value $length-cost))
             (let* (($CTV (recommendation-relational-ctv-with-support
                              $pattern $dbspace $support
                              $target-value $evidence-k))
                    ($organized (promote-engagement-conj $pattern)))
               (supportOf ($organized $CTV) $support))
             (empty)))
       (empty)))

(= (connected-materialize-typed
      (TypedPattern $placeholder-pattern $_bindings))
   (connected-materialize $placeholder-pattern))

;; The population and target counts are shared by every candidate. Compute
;; each pattern's joint and antecedent counts once for both MDL and its CTV.
(= (target-aware-formatter-cached
      $pattern $dbspace $min-support $target-value $evidence-k $length-cost
      $case-count $target-support)
   (if (recommendation-pattern-has-target? $pattern $target-value)
       (let* (($case-variable
                (recommendation-target-variable $pattern $target-value))
              ($joint
                (recommendation-distinct-support
                  $dbspace $pattern $case-variable)))
         (if (>= $joint $min-support)
             (let* (($antecedent
                      (recommendation-distinct-support
                        $dbspace (recommendation-antecedent $pattern)
                        $case-variable))
                    ((MDL $_data $_cost $net $_complexity)
                      (recommendation-mdl-from-counts
                        $joint $antecedent $case-count $target-support
                        (recommendation-pattern-complexity $pattern)
                        $length-cost)))
               (if (or (<= $length-cost 0.0) (> $net 0.0))
                   (supportOf
                     ((promote-engagement-conj $pattern)
                      (recommendation-ctv-from-counts
                        $joint $antecedent $case-count $target-support
                        $evidence-k))
                     $joint)
                   (empty)))
             (empty)))
       (empty)))

(= (closed-case-formatter
      $pattern $dbspace $min-support $target-value $evidence-k)
   (let $support (counter $dbspace $pattern)
     (if (>= $support $min-support)
         (let* (($CTV (recommendation-ctv-with-support
                         $pattern $dbspace $support
                         $target-value $evidence-k))
                ($organized (promote-engagement-conj $pattern)))
           (supportOf ($organized $CTV) $support))
         (empty))))

(= (closed-case-render-patterns
      $patterns $dbspace $min-support $target-value $evidence-k)
   (let $materialized
        (collapse
          (connected-materialize-typed (superpose $patterns)))
     (collapse
       (closed-case-formatter
         (superpose $materialized) $dbspace $min-support
         $target-value $evidence-k))))

(= (connected-render-patterns
      $patterns $dbspace $min-support
      $target-value $evidence-k $length-cost $required-functors)
   (let* (($materialized
            (alpha-unique-atom
              (collapse
                (let (TypedPattern $pattern $_bindings) (superpose $patterns)
                  (if (connected-has-required-functors?
                        $pattern
                        $required-functors)
                      (connected-materialize $pattern)
                      (empty))))))
          ($case-count (recommendation-case-count $dbspace))
          ($target-support (recommendation-target-support $dbspace $target-value)))
     (collapse
       (target-aware-formatter-cached
         (superpose $materialized) $dbspace $min-support
         $target-value $evidence-k $length-cost $case-count $target-support))))

;; schemas contain (predicate left-type right-type join|value|both) signatures.
;; They constrain joins and make specialization policy explicit.
(= (connected-frequency-pattern-miner
      $dbspace $min-support $depth $target-value $evidence-k
      $length-cost $schemas $max-patterns)
   (if (or (< $depth 2) (> $depth (size-atom (placeholder-names))))
       (MiningLimit depth $depth (size-atom (placeholder-names)))
       (let* (($templates
                (connected-templates
                  $dbspace $schemas $min-support))
              ($candidate-count (size-atom $templates)))
         (if (> $candidate-count $max-patterns)
             (MiningLimit candidates $candidate-count $max-patterns)
             (if (== $schemas ())
                 (let* (($patterns
                           (connected-value-patterns
                             $templates $depth $target-value))
                        ($pattern-count (size-atom $patterns)))
                   (if (> $pattern-count $max-patterns)
                       (MiningLimit patterns $pattern-count $max-patterns)
                       (closed-case-render-patterns
                         $patterns $dbspace $min-support $target-value
                         $evidence-k)))
                 (let* (($seed
                           ((TypedPattern
                              (, (engagement "$x" $target-value))
                              ((Binding "$x" exposure)))))
                        ($expanded
                          (connected-expand-to
                            $dbspace $seed $templates $min-support $target-value
                            1 $depth $max-patterns)))
                   (case $expanded
                     (((MiningLimit $kind $actual $limit)
                        (MiningLimit $kind $actual $limit))
                      ((Expanded $patterns)
                        (connected-render-patterns
                          $patterns $dbspace $min-support $target-value
                          $evidence-k $length-cost ()))))))))))

;; Mine all connected relational lengths in one expansion.  A target-only
;; support space may prune candidates because conjunction support is
;; anti-monotone; accepted rules are still calibrated and MDL-scored on the
;; complete population space.
(= (connected-frequency-pattern-miner-range-with-support
      $dbspace $supportspace $min-support $min-depth $max-depth
      $target-value $evidence-k $length-cost
      $schemas $required-functors $max-patterns)
   (if (or (== $schemas ())
           (or (< $min-depth 2)
               (or (> $min-depth $max-depth)
                   (> $max-depth (size-atom (placeholder-names))))))
       (MiningLimit depth-range $min-depth $max-depth)
       (let* (($templates
                (connected-templates $supportspace $schemas $min-support))
              ($candidate-count (size-atom $templates)))
         (if (> $candidate-count $max-patterns)
             (MiningLimit candidates $candidate-count $max-patterns)
             (if (connected-all-value-templates? $templates)
                 (let* (($patterns
                           (connected-frequent-value-patterns
                             $supportspace $templates $min-depth $max-depth
                             $min-support $target-value))
                        ($pattern-count (size-atom $patterns)))
                   (if (> $pattern-count $max-patterns)
                       (MiningLimit patterns $pattern-count $max-patterns)
                       (connected-render-patterns
                         $patterns $dbspace $min-support $target-value
                         $evidence-k $length-cost $required-functors)))
                 (let* (($seed
                           ((TypedPattern
                              (, (engagement "$x" $target-value))
                              ((Binding "$x" exposure)))))
                        ($expanded
                          (connected-expand-range
                            $supportspace $seed $templates
                            $min-support $target-value
                            1 $min-depth $max-depth () $max-patterns)))
                   (case $expanded
                     (((MiningLimit $kind $actual $limit)
                        (MiningLimit $kind $actual $limit))
                      ((Expanded $patterns)
                        (connected-render-patterns
                          $patterns $dbspace $min-support $target-value
                          $evidence-k $length-cost
                          $required-functors))))))))))

(= (connected-frequency-pattern-miner-range-prefiltered
      $dbspace $supportspace $min-support $min-depth $max-depth
      $target-value $evidence-k $length-cost
      $schemas $required-functors $max-patterns)
   (connected-frequency-pattern-miner-range-with-support
     $dbspace $supportspace $min-support $min-depth $max-depth
     $target-value $evidence-k $length-cost
     $schemas $required-functors $max-patterns))

(= (connected-frequency-pattern-miner-range
      $dbspace $min-support $min-depth $max-depth
      $target-value $evidence-k $length-cost
      $schemas $required-functors $max-patterns)
   (connected-frequency-pattern-miner-range-with-support
     $dbspace $dbspace $min-support $min-depth $max-depth
     $target-value $evidence-k $length-cost
     $schemas $required-functors $max-patterns))

;; Existing callers use the same expansion engine without multi-variable join
;; predicates or a length penalty. Their target-aware outputs remain unchanged.
(= (frequency-pattern-miner
      $dbspace $min-support $depth $target-value $evidence-k)
   (connected-frequency-pattern-miner
     $dbspace $min-support $depth $target-value $evidence-k
     0.0 () 250000))

(= (frequency-pattern-miner $dbspace $min-support $depth)
   (frequency-pattern-miner
     $dbspace $min-support $depth "click" 800.0))
