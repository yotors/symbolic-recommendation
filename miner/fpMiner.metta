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
      $target-value $evidence-k $length-cost)
   (let $materialized
        (alpha-unique-atom
          (collapse
            (connected-materialize-typed (superpose $patterns))))
     (collapse
       (target-aware-formatter
         (superpose $materialized) $dbspace $min-support
         $target-value $evidence-k $length-cost))))

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
                          $evidence-k $length-cost))))))))))

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
