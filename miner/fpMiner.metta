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
