;; Read the parent MDL bound at ranking time; -1 always excludes an exactly
;; redundant child. A negative cutoff disables the parent filter entirely.
(= (adaptive-parent-votes $space $case $cutoff)
   (collapse
     (match $space
       (, (RuleFired $case $rule)
          (RuleMeta $rule $depth $quality $strength $confidence $bound $_families))
       (if (or (< $cutoff 0.0) (> $bound $cutoff))
           (Vote $depth $quality $strength $confidence)
           (empty)))))

(= (adaptive-parent-score $space $case $cutoff $base $mode)
   (let $votes (adaptive-parent-votes $space $case $cutoff)
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

(= (adaptive-parent-scores $space $cutoff $base $mode)
   (collapse
     (match $space (CandidateCase $case)
       (PredicateScore $case
         (adaptive-parent-score $space $case $cutoff $base $mode)))))
