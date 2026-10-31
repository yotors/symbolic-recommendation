(= (required-functors) (engagement))
(= (placeholder-names) ("$x" "$y" "$z" "$w" "$v" "$u"))

(= (functor-of $e) (car-atom $e))
(= (replace-key $term $key $var)
   (if (is-expr $term)
       (replace-key-list $term $key $var)
       (if (== $term $key) $var $term)))

(= (replace-key-list $ls $key $var)
   (if (== $ls ())
       ()
       (let ($h $t) (decons-atom $ls)
            (cons-atom (replace-key $h $key $var)
                       (replace-key-list $t $key $var)))))

(= (promote-placeholders $term $names)
   (if (== $names ())
       $term
       (let* ((($name $rest) (decons-atom $names))
              ($var (parse $name)))
         (promote-placeholders (replace-key $term $name $var) $rest))))

(= (promote-engagement-conj $conjunction)
   (let ($others $required) (split-required (cdr-atom $conjunction))
        (append $others $required)))

(= (split-required $cs)
   (if (== $cs ())
       (() ())
       (let* ((($h $t) (decons-atom $cs))
              (($others $required) (split-required $t)))
         (if (is-member (functor-of $h) (required-functors))
             ($others (cons-atom $h $required))
             ((cons-atom $h $others) $required)))))

(= (counter $db $ptrn)
    (let $got (collapse (match $db $ptrn 1)) (size-atom $got)))

;; Recommendation support is measured in distinct exposure/case identifiers.
;; A relational conjunction may have several valid graph bindings for the same
;; exposure; those paths are evidence provenance, not independent outcomes.
(= (recommendation-target-variable-in () $_target-value) (empty))
(= (recommendation-target-variable-in $clauses $target-value)
   (let ($head $tail) (decons-atom $clauses)
     (case $head
       (((engagement $case $value)
          (if (== $value $target-value)
              $case
              (recommendation-target-variable-in $tail $target-value)))
        ($_other
          (recommendation-target-variable-in $tail $target-value))))))

(= (recommendation-target-variable $pattern $target-value)
   (let $clauses (if (== (car-atom $pattern) ,)
                     (cdr-atom $pattern)
                     ($pattern))
     (recommendation-target-variable-in $clauses $target-value)))

(= (recommendation-distinct-support $db $pattern $case-variable)
   (size-atom
     (unique-atom
       (collapse (match $db $pattern $case-variable)))))

(= (recommendation-pattern-support $db $pattern $target-value)
   (recommendation-distinct-support
     $db $pattern
     (recommendation-target-variable $pattern $target-value)))

;; Remove the outcome from a mined comma-conjunction and keep every premise.
;; This works for both pair mining (one feature -> outcome) and deeper mining
;; (multiple features -> outcome), so antecedent support is always nA.
(= (recommendation-engagement? $atom)
   (case $atom
     (((engagement $x $y) True)
      ($other False))))

(= (recommendation-remove-engagement ()) ())
(= (recommendation-remove-engagement $clauses)
   (let ($head $tail) (decons-atom $clauses)
     (if (recommendation-engagement? $head)
         (recommendation-remove-engagement $tail)
         (cons-atom $head (recommendation-remove-engagement $tail)))))

(= (recommendation-wrap-antecedent $clauses)
   (if (> (size-atom $clauses) 1)
       (cons-atom , $clauses)
       (car-atom $clauses)))

(= (recommendation-antecedent $atoms)
   (let $clauses (if (== (car-atom $atoms) ,) (cdr-atom $atoms) ($atoms))
     (recommendation-wrap-antecedent
       (recommendation-remove-engagement $clauses))))

;; Target-aware CTV helpers for the recommendation scratch-space contract.
;; Each closed case has exactly one engagement label.  The universe is the
;; number of distinct case identifiers, never the number of feature atoms.
(= (recommendation-target-clause? $atom $target-value)
   (case $atom
     (((engagement $_case $value) (== $value $target-value))
      ($_other False))))

(= (recommendation-has-target-value? () $_target-value) False)
(= (recommendation-has-target-value? $clauses $target-value)
   (let ($head $tail) (decons-atom $clauses)
     (if (recommendation-target-clause? $head $target-value)
         True
         (recommendation-has-target-value? $tail $target-value))))

(= (recommendation-pattern-has-target? $pattern $target-value)
   (let $clauses (if (== (car-atom $pattern) ,)
                     (cdr-atom $pattern)
                     ($pattern))
     (recommendation-has-target-value? $clauses $target-value)))

(= (recommendation-case-count $db)
   (size-atom
     (unique-atom
       (collapse
         (match $db (engagement $case $_value) $case)))))

(= (recommendation-target-support $db $target-value)
   (size-atom
     (unique-atom
       (collapse
         (match $db (engagement $case $target-value) $case)))))

(= (recommendation-count-confidence $count $evidence-k)
   (if (<= $evidence-k 0)
       (empty)
       (if (<= $count 0)
           0.0
           (// $count (+ $count $evidence-k)))))

;; Build PeTTa's positive and complement branches from one exact contingency.
;; Keeping this operation in MeTTa makes the same count semantics available to
;; discovery, complete-population calibration and future delta updates.
(= (recommendation-ctv-from-counts
      $joint-support $antecedent-support $case-count $target-support $evidence-k)
   (let* (($outside-support (- $case-count $antecedent-support))
          ($outside-target (- $target-support $joint-support))
          ($valid
             (and (>= $joint-support 0)
               (and (<= $joint-support $antecedent-support)
                 (and (<= $joint-support $target-support)
                   (and (>= $outside-support 0)
                     (and (>= $outside-target 0)
                          (<= $outside-target $outside-support))))))))
     (if $valid
         (CTV
           (STV
             (if (<= $antecedent-support 0)
                 0.0
                 (// $joint-support $antecedent-support))
             (recommendation-count-confidence
               $antecedent-support $evidence-k))
           (STV
             (if (<= $outside-support 0)
                 0.0
                 (// $outside-target $outside-support))
             (recommendation-count-confidence $outside-support $evidence-k)))
         (empty))))

;; Convert the validated contingency (nAY, nA, N, nY) into PeTTa's two
;; conditional branches.  These discovery values are audit evidence only;
;; complete-population calibration replaces them before rule compilation.
(= (recommendation-ctv-with-support
      $pattern $db $joint-support $target-value $evidence-k)
   (let* (($antecedent-support
              (counter $db (recommendation-antecedent $pattern)))
          ($case-count (recommendation-case-count $db))
          ($target-support (recommendation-target-support $db $target-value)))
     (recommendation-ctv-from-counts
       $joint-support $antecedent-support $case-count $target-support $evidence-k)))

(= (recommendation-relational-ctv-with-support
      $pattern $db $joint-support $target-value $evidence-k)
   (let* (($case-variable
              (recommendation-target-variable $pattern $target-value))
          ($antecedent-support
              (recommendation-distinct-support
                $db (recommendation-antecedent $pattern) $case-variable))
          ($case-count (recommendation-case-count $db))
          ($target-support (recommendation-target-support $db $target-value)))
     (recommendation-ctv-from-counts
       $joint-support $antecedent-support $case-count $target-support $evidence-k)))

;; Minimum-description-length gate for relational rules.  The data term is
;; the Bernoulli log-loss saved by splitting the population on the antecedent.
;; The model term charges each premise and every join variable beyond the
;; exposure variable.  length-cost=0 disables the gate exactly.
(= (recommendation-binomial-nll $positive $total)
   (if (or (<= $total 0)
           (or (<= $positive 0) (>= $positive $total)))
       0.0
       (let* (($negative (- $total $positive))
              ($p (// $positive $total))
              ($q (// $negative $total)))
         (- 0.0
            (+ (* $positive (log-math 2.718281828459045 $p))
               (* $negative (log-math 2.718281828459045 $q)))))))

(= (recommendation-collect-variables $term)
   (if (is-expr $term)
       (recommendation-collect-variable-list $term)
       (if (== (get-metatype $term) Variable) ($term) ())))

(= (recommendation-collect-variable-list $terms)
   (if (== $terms ())
       ()
       (let ($head $tail) (decons-atom $terms)
         (union-atom
           (recommendation-collect-variables $head)
           (recommendation-collect-variable-list $tail)))))

(= (recommendation-pattern-specificity $pattern)
   (size-atom
     (let $clauses (if (== (car-atom $pattern) ,)
                       (cdr-atom $pattern)
                       ($pattern))
       (recommendation-remove-engagement $clauses))))

(= (recommendation-pattern-complexity $pattern)
   (let* (($premises (recommendation-pattern-specificity $pattern))
          ($variables
            (size-atom
              (unique-atom
                (recommendation-collect-variables $pattern)))))
     (+ $premises (max 0 (- $variables 1)))))

(= (recommendation-mdl-from-counts
      $joint-support $antecedent-support $case-count $target-support
      $complexity $length-cost)
   (let* (($outside-support (- $case-count $antecedent-support))
          ($outside-target (- $target-support $joint-support))
          ($base-nll
            (recommendation-binomial-nll $target-support $case-count))
          ($split-nll
            (+ (recommendation-binomial-nll
                 $joint-support $antecedent-support)
               (recommendation-binomial-nll
                 $outside-target $outside-support)))
          ($data-gain (- $base-nll $split-nll))
          ($model-cost
            (* $length-cost
               (* $complexity
                  (log-math 2.718281828459045 (+ 1 $case-count)))))
          ($net-gain (- $data-gain $model-cost)))
     (MDL $data-gain $model-cost $net-gain $complexity)))

(= (recommendation-mdl-score
      $pattern $db $target-value $length-cost)
   (let* (($case-variable
            (recommendation-target-variable $pattern $target-value))
          ($joint-support
            (recommendation-distinct-support $db $pattern $case-variable))
          ($antecedent-support
            (recommendation-distinct-support
              $db (recommendation-antecedent $pattern) $case-variable))
          ($case-count (recommendation-case-count $db))
          ($target-support
            (recommendation-target-support $db $target-value)))
     (recommendation-mdl-from-counts
       $joint-support $antecedent-support $case-count $target-support
       (recommendation-pattern-complexity $pattern) $length-cost)))

(= (recommendation-mdl-accepts?
      $pattern $db $target-value $length-cost)
   (if (<= $length-cost 0.0)
       True
       (let (MDL $_data $_cost $net $_complexity)
            (recommendation-mdl-score
              $pattern $db $target-value $length-cost)
         (> $net 0.0))))

;; Render one calibrated rule from sufficient statistics.  Both full and delta
;; calibration terminate here, so their CTV and selection-quality semantics
;; cannot drift apart.
(= (recommendation-calibration-from-counts
      $rule-id $joint-support $antecedent-support $case-count
      $target-support $specificity $evidence-k)
   (let* (
          ($base-rate (if (<= $case-count 0)
                          0.0
                          (// $target-support $case-count)))
          ($strength (if (<= $antecedent-support 0)
                         0.0
                         (// $joint-support $antecedent-support)))
          ($confidence
             (recommendation-count-confidence $antecedent-support $evidence-k))
          ($lift (if (<= $base-rate 0.0) 0.0 (// $strength $base-rate)))
          ($quality
             (* $confidence
                (* (abs-math (- $strength $base-rate))
                   (* (+ 1.0 (* 0.25 (- $specificity 1)))
                      (log-math 2.718281828459045
                                (+ 1 $antecedent-support))))))
          ($CTV (recommendation-ctv-from-counts
                  $joint-support $antecedent-support $case-count
                  $target-support $evidence-k)))
     (PointCalibration
       $rule-id $CTV
       (Counts $joint-support $antecedent-support $case-count $target-support)
       (Metrics $base-rate $lift $specificity $quality))))

;; Re-estimate one discovered rule over the complete retained population.
(= (recommendation-calibrate-rule
      $rule-id $pattern $db $target-value $evidence-k)
   (let* (($joint-support (counter $db $pattern))
          ($antecedent (recommendation-antecedent $pattern))
          ($antecedent-support (counter $db $antecedent))
          ($case-count (recommendation-case-count $db))
          ($target-support (recommendation-target-support $db $target-value))
          ($specificity
             (size-atom
               (let $clauses (if (== (car-atom $pattern) ,)
                                 (cdr-atom $pattern)
                                 ($pattern))
                 (recommendation-remove-engagement $clauses)))))
     (recommendation-calibration-from-counts
       $rule-id $joint-support $antecedent-support $case-count
       $target-support $specificity $evidence-k)))

;; Add exact counts from an append-only population delta to a prior snapshot.
;; The host supplies old counts as inert data; every update and derived numeric
;; value is evaluated here in MeTTa.
(= (recommendation-update-calibration
      $rule-id $pattern $delta-db $target-value $evidence-k
      $old-joint $old-antecedent $old-cases $old-target)
   (let* (($delta-joint (counter $delta-db $pattern))
          ($antecedent (recommendation-antecedent $pattern))
          ($delta-antecedent (counter $delta-db $antecedent))
          ($delta-cases (recommendation-case-count $delta-db))
          ($delta-target
             (recommendation-target-support $delta-db $target-value))
          ($specificity
             (size-atom
               (let $clauses (if (== (car-atom $pattern) ,)
                                 (cdr-atom $pattern)
                                 ($pattern))
                 (recommendation-remove-engagement $clauses)))))
     (recommendation-calibration-from-counts
       $rule-id
       (+ $old-joint $delta-joint)
       (+ $old-antecedent $delta-antecedent)
       (+ $old-cases $delta-cases)
       (+ $old-target $delta-target)
       $specificity $evidence-k)))

(= (sup-eval $db $pattern $minSup)
    (let $matches (collapse (match $db $pattern $pattern))
        (>= (size-atom $matches) $minSup)
    ))

;; Match PeTTaChainer's evidence confidence model: count/(count + 800).
(= (Default_k) 800.0)
(= (// $a $b) (/ (* 1.0 $a) $b))
(= (count_to_confidence $x) (// $x (+ $x (Default_k))))

(= (emp-tv $pattern $db)
    (let*
        (
            ($antecedent (antecedent $pattern))
            ( $sup-ante (sup-num $db $antecedent))
            ( $sup (sup-num $db $pattern))
            ( $conf (count_to_confidence $sup))
            ($mean (if (<= $sup-ante 0) 0.0 (// $sup $sup-ante)))
        )
    (STV $mean $conf)))
