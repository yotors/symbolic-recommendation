;; Rank with a single best proven rule; this avoids treating overlapping
;; conjunctions as independent evidence. Optional parent and family filters
;; are applied before selecting the deepest rule and then strongest quality.
(= (predicate-rule-quality $strength $confidence $support $depth $base)
   (* (* $confidence (abs-math (- $strength $base)))
      (* (log-math 2.718281828459045 (+ 1 $support))
         (+ 1 (* 0.1 (- $depth 1))))))

(= (predicate-votes $space $case $parent-only $excluded)
   (collapse
     (match $space
       (, (RuleFired $case $rule)
          (RuleMeta $rule $depth $quality $strength $confidence $parent $families))
       (if (and (or (== $parent-only False) (== $parent True))
                (not (is-member $excluded $families)))
           (Vote $depth $quality $strength $confidence)
           (empty)))))

(= (predicate-vote-depth (Vote $depth $_quality $_strength $_confidence)) $depth)
(= (predicate-depth-quality (Vote $depth $quality $_strength $_confidence) $wanted)
   (if (== $depth $wanted) $quality (empty)))

(= (predicate-vote-probability
      (Vote $depth $quality $strength $confidence)
      $wanted-depth $wanted-quality $base $mode)
   (if (and (== $depth $wanted-depth) (== $quality $wanted-quality))
       (if (== $mode raw)
           $strength
           (+ $base (* $confidence (- $strength $base))))
       (empty)))

(= (predicate-rank-score $space $case $parent-only $excluded $base $mode)
   (let $votes (predicate-votes $space $case $parent-only $excluded)
     (if (== $votes ())
         $base
         (let* (($depth
                  (max-atom
                    (collapse (predicate-vote-depth (superpose $votes)))))
                ($quality
                  (max-atom
                    (collapse
                      (predicate-depth-quality (superpose $votes) $depth)))))
           (max-atom
             (collapse
               (predicate-vote-probability
                 (superpose $votes) $depth $quality $base $mode)))))))

(= (predicate-rank-scores $space $parent-only $excluded $base $mode)
   (collapse
     (match $space (CandidateCase $case)
       (PredicateScore $case
         (predicate-rank-score $space $case $parent-only $excluded $base $mode)))))
