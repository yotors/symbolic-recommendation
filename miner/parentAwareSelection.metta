;; Test whether a conjunction improves prediction within every immediate parent.
;; The evidence for a child is compared with the rest of its parent, avoiding
;; the global-null test that accepts redundant extensions.
(= (parent-removals ()) (empty))
(= (parent-removals $clauses)
   (let ($head $tail) (decons-atom $clauses)
     (superpose
       ($tail
        (let $rest (parent-removals $tail)
          (cons-atom $head $rest))))))

(= (parent-incremental-gain
      $db $child $parent-clauses $target $length-cost)
   (let* (($parent
             (cons-atom ,
               (append $parent-clauses ($target))))
          ($target-value (car-atom (cdr-atom (cdr-atom $target))))
          ($case-variable (recommendation-target-variable $child $target-value))
          ($child-n (recommendation-distinct-support
                      $db (recommendation-antecedent $child) $case-variable))
          ($child-y (recommendation-pattern-support $db $child $target-value))
          ($parent-n (recommendation-distinct-support
                       $db (recommendation-antecedent $parent) $case-variable))
          ($parent-y (recommendation-pattern-support $db $parent $target-value))
          ($residual-n (- $parent-n $child-n))
          ($residual-y (- $parent-y $child-y)))
     (if (<= $residual-n 0)
         -1.0
         (- (- (recommendation-binomial-nll $parent-y $parent-n)
               (+ (recommendation-binomial-nll $child-y $child-n)
                  (recommendation-binomial-nll $residual-y $residual-n)))
            (* $length-cost
               (log-math 2.718281828459045 (+ 1 $parent-n)))))))

(= (parent-minimum-gain $db $child $target $length-cost)
   (let $clauses (recommendation-remove-engagement (cdr-atom $child))
     (if (<= (size-atom $clauses) 1)
         (let (MDL $_data $_cost $net $_complexity)
              (recommendation-mdl-score
                $child $db
                (car-atom (cdr-atom (cdr-atom $target))) $length-cost)
           $net)
         (min-atom
           (collapse
             (let $parent (parent-removals $clauses)
               (parent-incremental-gain
                 $db $child $parent $target $length-cost)))))))

;; A child must add information relative to every immediate parent. Return
;; the smallest gain/log(parent support + 1), so changing a nonnegative MDL
;; cost needs no repeated support scans. -1 denotes identical coverage.
(= (parent-penalty-bound
      $db $parent-clauses $target $case-variable $target-value
      $child-n $child-y)
   (let* (($parent (cons-atom , (append $parent-clauses ($target))))
          ($parent-n (recommendation-distinct-support
                       $db (recommendation-antecedent $parent) $case-variable))
          ($parent-y (recommendation-pattern-support $db $parent $target-value))
          ($residual-n (- $parent-n $child-n))
          ($residual-y (- $parent-y $child-y)))
     (if (<= $residual-n 0)
         -1.0
         (let $gain
              (- (recommendation-binomial-nll $parent-y $parent-n)
                 (+ (recommendation-binomial-nll $child-y $child-n)
                    (recommendation-binomial-nll $residual-y $residual-n)))
           (/ (max 0.0 $gain)
              (log-math 2.718281828459045 (+ 1 $parent-n)))))))

(= (parent-minimum-penalty-bound $db $child $target)
   (let $clauses (recommendation-remove-engagement (cdr-atom $child))
     (if (<= (size-atom $clauses) 1)
         1000000000.0
         (let* (($target-value (car-atom (cdr-atom (cdr-atom $target))))
                ($case-variable (recommendation-target-variable $child $target-value))
                ($child-n (recommendation-distinct-support
                            $db (recommendation-antecedent $child) $case-variable))
                ($child-y (recommendation-pattern-support $db $child $target-value)))
           (min-atom
             (collapse
               (let $parent (parent-removals $clauses)
                 (parent-penalty-bound
                   $db $parent $target $case-variable $target-value
                   $child-n $child-y))))))))
