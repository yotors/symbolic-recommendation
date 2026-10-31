;; Target-aware connected conjunction expansion over binary facts.
;; Predicate signatures prevent invalid joins, and the exposure type is an
;; anchor: expansion may reuse it but never introduce another exposure.

(= (connected-clauses $pattern) (cdr-atom $pattern))

(= (connected-has-functor? () $_functor) False)
(= (connected-has-functor? $clauses $functor)
   (let ($head $tail) (decons-atom $clauses)
     (if (== (functor-of $head) $functor)
         True
         (connected-has-functor? $tail $functor))))

(= (connected-has-required-functors? $pattern $required)
   (if (== $required ())
       True
       (let ($head $tail) (decons-atom $required)
         (if (connected-has-functor?
               (connected-clauses $pattern) $head)
             (connected-has-required-functors? $pattern $tail)
             False))))

(= (connected-first-unused () $_used) (empty))
(= (connected-first-unused $available $used)
   (let ($head $tail) (decons-atom $available)
     (if (is-member $head $used)
         (connected-first-unused $tail $used)
         $head)))

(= (connected-binding-variable (Binding $variable $_type)) $variable)
(= (connected-binding-type (Binding $_variable $type)) $type)

(= (connected-binding-variables $bindings)
   (collapse
     (connected-binding-variable (superpose $bindings))))

(= (connected-variables-of-type $bindings $type)
   (collapse
     (let $binding (superpose $bindings)
       (if (== (connected-binding-type $binding) $type)
           (connected-binding-variable $binding)
           (empty)))))

(= (connected-add-binding $bindings $variable $type)
   (append $bindings ((Binding $variable $type))))

(= (connected-materialize $pattern)
   (promote-placeholders $pattern (placeholder-names)))

(= (connected-bind-value-template
      (ValueTemplate $template $left-type) $bindings)
   (let $variables (connected-variables-of-type $bindings $left-type)
     (let $left (superpose $variables)
       (Bound (replace-key $template "@left" $left) $bindings))))

(= (connected-bind-join-template
      (JoinTemplate $template $left-type $right-type) $bindings)
   (let* (($left-existing
            (connected-variables-of-type $bindings $left-type))
          ($right-existing
            (connected-variables-of-type $bindings $right-type))
          ($used (connected-binding-variables $bindings))
          ($fresh (connected-first-unused (placeholder-names) $used)))
     (superpose
       ((let* (($left (superpose $left-existing))
               ($right (superpose $right-existing)))
          (Bound
            (replace-key
              (replace-key $template "@left" $left)
              "@right" $right)
            $bindings))
        (if (or (== $right-type exposure)
                (not (== $right-existing ())))
            (empty)
            (let $left (superpose $left-existing)
              (Bound
                (replace-key
                  (replace-key $template "@left" $left)
                  "@right" $fresh)
                (connected-add-binding
                  $bindings $fresh $right-type))))))))

(= (connected-bind-template $template $bindings)
   (case $template
     (((ValueTemplate $_atom $_left-type)
        (connected-bind-value-template $template $bindings))
      ((JoinTemplate $_atom $_left-type $_right-type)
        (connected-bind-join-template $template $bindings)))))

(= (connected-template-functor $template)
   (case $template
     (((ValueTemplate $atom $_left-type) (functor-of $atom))
      ((JoinTemplate $atom $_left-type $_right-type) (functor-of $atom)))))

(= (connected-add-template
      $db (TypedPattern $pattern $bindings) $template
      $min-support $target-value)
   (let $clauses (connected-clauses $pattern)
     (if (connected-has-functor?
           $clauses (connected-template-functor $template))
         (empty)
         (let* (((Bound $clause $next-bindings)
                   (connected-bind-template $template $bindings))
                ($expanded
                  (cons-atom ,
                    (sort-atom (append $clauses ($clause)))))
                ($materialized (connected-materialize $expanded))
                ($support
                  (recommendation-pattern-support
                    $db $materialized $target-value)))
           (if (>= $support $min-support)
               (TypedPattern $expanded $next-bindings)
               (empty))))))

(= (connected-expand-step
      $db $patterns $templates $min-support $target-value)
   (unique-atom
     (collapse
       (connected-add-template
         $db (superpose $patterns) (superpose $templates)
         $min-support $target-value))))

(= (connected-expand-to
      $db $patterns $templates $min-support $target-value
      $current-depth $max-depth $max-patterns)
   (if (>= $current-depth $max-depth)
       (Expanded $patterns)
       (let* (($next
                (connected-expand-step
                  $db $patterns $templates $min-support $target-value))
              ($count (size-atom $next)))
         (if (> $count $max-patterns)
             (MiningLimit patterns $count $max-patterns)
             (connected-expand-to
               $db $next $templates $min-support $target-value
               (+ $current-depth 1) $max-depth $max-patterns)))))

;; Expand once and retain every requested conjunction length.  The common
;; pattern budget bounds the whole frontier, not each depth independently.
(= (connected-expand-range
      $db $frontier $templates $min-support $target-value
      $current-depth $min-depth $max-depth $retained $max-patterns)
   (if (>= $current-depth $max-depth)
       (Expanded $retained)
       (let* (($next
                (connected-expand-step
                  $db $frontier $templates $min-support $target-value))
              ($next-depth (+ $current-depth 1))
              ($next-retained
                (if (>= $next-depth $min-depth)
                    (unique-atom (append $retained $next))
                    $retained))
              ($count (+ (size-atom $next) (size-atom $next-retained))))
         (if (> $count $max-patterns)
             (MiningLimit patterns $count $max-patterns)
             (connected-expand-range
               $db $next $templates $min-support $target-value
               $next-depth $min-depth $max-depth
               $next-retained $max-patterns)))))

(= (connected-value-template
      $db $functor $value $left-type $min-support)
   (if (and (not (== $functor engagement))
            (>= (counter $db ($functor $_left $value)) $min-support))
       (ValueTemplate ($functor "@left" $value) $left-type)
       (empty)))

(= (connected-default-value-templates $db $min-support)
   (alpha-unique-atom
     (collapse
       (match $db ($functor $_left $value)
         (connected-value-template
           $db $functor $value exposure $min-support)))))

(= (connected-schema-value-templates $db $schemas $min-support)
   (alpha-unique-atom
     (collapse
       (let ($functor $left-type $_right-type $mode) (superpose $schemas)
         (if (or (== $mode value) (== $mode both))
             (match $db ($functor $_left $value)
               (connected-value-template
                 $db $functor $value $left-type $min-support))
             (empty))))))

(= (connected-join-template
      $db $functor $left-type $right-type $min-support)
   (if (and (not (== $functor engagement))
            (>= (counter $db ($functor $_left $_right)) $min-support))
       (JoinTemplate
         ($functor "@left" "@right") $left-type $right-type)
       (empty)))

(= (connected-join-templates $db $schemas $min-support)
   (alpha-unique-atom
     (collapse
       (let ($functor $left-type $right-type $mode) (superpose $schemas)
         (if (or (== $mode join) (== $mode both))
             (connected-join-template
               $db $functor $left-type $right-type $min-support)
             (empty))))))

(= (connected-templates $db $schemas $min-support)
   (if (== $schemas ())
       (connected-default-value-templates $db $min-support)
       (unique-atom
         (append
           (connected-schema-value-templates $db $schemas $min-support)
           (connected-join-templates $db $schemas $min-support)))))

;; Closed-case workspaces contain value templates sharing the exposure only.
;; Choose each unordered predicate combination once instead of rediscovering
;; every permutation through graph expansion.
(= (connected-value-template-clause
      (ValueTemplate $template $_left-type))
   (replace-key $template "@left" "$x"))

(= (connected-choose-values
      $templates $remaining $selected $used-functors $target-value)
   (if (== $remaining 0)
       (TypedPattern
         (cons-atom ,
           (sort-atom
             (append $selected ((engagement "$x" $target-value)))))
         ((Binding "$x" exposure)))
       (if (== $templates ())
           (empty)
           (let* ((($head $tail) (decons-atom $templates))
                  ($functor (connected-template-functor $head)))
             (superpose
               ((if (is-member $functor $used-functors)
                    (empty)
                    (connected-choose-values
                      $tail (- $remaining 1)
                      (append
                        $selected
                        ((connected-value-template-clause $head)))
                      (append $used-functors ($functor)) $target-value))
                (connected-choose-values
                  $tail $remaining $selected $used-functors
                  $target-value)))))))

(= (connected-value-patterns $templates $depth $target-value)
   (unique-atom
     (collapse
       (connected-choose-values
         $templates (- $depth 1) () () $target-value))))
