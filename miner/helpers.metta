(= (required-functors) (engagement))
(= (placeholder-names) ("$x" "$y"))

(= (clause-placeholders $term)
   (unique-atom (collect-placeholders $term)))

(= (collect-placeholders $term)
   (if (is-expr $term)
       (collect-placeholders-list $term)
       (if (is-member $term (placeholder-names)) ($term) ())))

(= (collect-placeholders-list $l)
   (if (== $l ())
       ()
       (let ($h $t) (decons-atom $l)
            (union-atom (collect-placeholders $h)
                        (collect-placeholders-list $t)))))
(= (functor-of $e) (car-atom $e))

(= (build-infos $expr)
      ($expr (clause-placeholders $expr) (functor-of $expr)))

(= (collect-hubs $infos) (unique-atom (all-placeholders $infos)))

(= (all-placeholders $infos)
   (if (== $infos ())
       ()
       (let (($e $ks $f) $t) (decons-atom $infos)
            (union-atom $ks (all-placeholders $t)))))

(= (pool-with-hub ($e $ks $f) $hub)
   (if (is-member $hub $ks) ($e $ks $f) (empty)))
(= (choose-combo $pool $plen $hub $k $sel $used-functors $used-placeholders $missing)
   (if (== $k 0)
       (if (== $missing ())
           ((reverse $sel) (cons-atom $hub $used-placeholders))
           (empty))
       (if (< $plen $k)
           (empty)
           (let* ((($info $rest) (decons-atom $pool))
                  ($rlen (- $plen 1)))
             (if (can-add $hub $info $used-functors $used-placeholders)
                 (let* ((($e $ks $f) $info)
                        ($non-hub (exclude-item $hub $ks))
                        ($functors2 (cons-atom $f $used-functors))
                        ($placeholders2 (union-atom $non-hub $used-placeholders))
                        ($missing2 (exclude-item $f $missing)))
                   (superpose
                      ((choose-combo $rest $rlen $hub (- $k 1)
                                     (cons-atom $e $sel) $functors2
                                     $placeholders2 $missing2)
                       (choose-combo $rest $rlen $hub $k $sel
                                     $used-functors $used-placeholders $missing))))
                 (choose-combo $rest $rlen $hub $k $sel
                               $used-functors $used-placeholders $missing))))))

(= (can-add $hub ($e $ks $f) $used-functors $used-placeholders)
   (if (not (is-member $f $used-functors))
       (== (intersection-atom
              (exclude-item $hub $ks)
              $used-placeholders)
           ())
       False))
(= (generate-combo $infos $hubs $k)
   (let $hub (superpose $hubs)
     (let* (($pool (collapse (pool-with-hub (superpose $infos) $hub)))
            ($plen (size-atom $pool)))
       (if (and (>= $plen $k) (pool-covers-required $pool))
           (choose-combo $pool $plen $hub $k () () () (required-functors))
           (empty)))))

(= (pool-covers-required $pool)
   (subset-of (required-functors) (info-functors $pool)))

(= (info-functors $infos)
   (collapse (let ($e $ks $f) (superpose $infos) $f)))

(= (subset-of $xs $ys)
   (if (== $xs ())
       True
       (let ($h $t) (decons-atom $xs)
            (if (is-member $h $ys) (subset-of $t $ys) False))))
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

(= (finalize ($combo $names))
   (conjunct (cons-atom , (promote-placeholders $combo $names))))

(= (unique-combinations-star $exprs $size)
   (let $n (size-atom $exprs)
        (if (<= $size 0)
            ()
            (if (> $size $n)
                ()
                (let* (($infos (collapse (build-infos (superpose $exprs))))
                       ($raw (collapse
                                (generate-combo $infos (collect-hubs $infos) $size))))
                  (collapse (finalize (superpose $raw))))))))
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

;; Convert the validated contingency (nAY, nA, N, nY) into PeTTa's two
;; conditional branches.  These discovery values are audit evidence only;
;; The application recalibrates every retained structure on the full causal population.
(= (recommendation-ctv-with-support
      $pattern $db $joint-support $target-value $evidence-k)
   (let* (($antecedent-support
              (counter $db (recommendation-antecedent $pattern)))
          ($case-count (recommendation-case-count $db))
          ($target-support (recommendation-target-support $db $target-value))
          ($outside-support (- $case-count $antecedent-support))
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
