;; Hierarchical backoff over proven derived-feature conjunctions.
;; Prefer the deepest available proof, then the strongest precomputed rule
;; quality at that depth.  Numerical score aggregation stays inside PeTTa.

(= (derived-proof-depths $space $case $max-depth)
   (collapse
     (match $space
       (RuleEvidence $case $_rule $depth $_quality $_posterior)
       (if (<= $depth $max-depth) $depth (empty)))))

(= (derived-depth-qualities $space $case $depth)
   (collapse
     (match $space
       (RuleEvidence $case $_rule $depth $quality $_posterior)
       $quality)))

(= (derived-selected-posteriors $space $case $depth $quality)
   (collapse
     (match $space
       (RuleEvidence $case $_rule $depth $quality $posterior)
       $posterior)))

(= (derived-backoff-score $space $case $max-depth $base-rate)
   (let $depths (derived-proof-depths $space $case $max-depth)
     (if (== $depths ())
         $base-rate
         (let* (($depth (max-atom $depths))
                ($quality
                  (max-atom
                    (derived-depth-qualities $space $case $depth))))
           (max-atom
             (derived-selected-posteriors
               $space $case $depth $quality))))))

(= (derived-case-scores $space $max-depth $base-rate)
   (collapse
     (match $space
       (CandidateCase $case)
       (DerivedScore
         $case
         (derived-backoff-score $space $case $max-depth $base-rate)))))
