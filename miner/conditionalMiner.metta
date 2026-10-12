;; Target-aware expansion of fpMiner seeds.
;; Compact case rows are only the retrieval index. PeTTa computes the
;; sufficient statistics, temporal stability, rule selection, and scores.

(= (conditional-observation-step
      (ObservationMass $scope $target $positive $negative
                       $positive-count $negative-count)
      (Observation $fold $outcome $weight))
   (if (or (== $scope AllFolds) (== $scope $fold))
       (if (== $outcome $target)
           (ObservationMass $scope $target
             (+ $positive $weight) $negative
             (+ $positive-count 1) $negative-count)
           (ObservationMass $scope $target
             $positive (+ $negative $weight)
             $positive-count (+ $negative-count 1)))
       (ObservationMass $scope $target $positive $negative
                        $positive-count $negative-count)))

(= (conditional-observation-mass $observations $scope $target)
   (let (ObservationMass $_scope $_target
          $positive $negative $positive-count $negative-count)
        (foldl-atom $observations
          (ObservationMass $scope $target 0.0 0.0 0 0)
          conditional-observation-step)
     (Mass $positive $negative $positive-count $negative-count)))

(= (conditional-fold-cells () $_observations $_key $_target) ())
(= (conditional-fold-cells $folds $observations $key $target)
   (let ($fold $tail) (decons-atom $folds)
     (cons-atom
       (Cell $key $fold
         (conditional-observation-mass $observations $fold $target))
       (conditional-fold-cells $tail $observations $key $target))))

(= (conditional-column-values $db $pattern $placeholders)
   (let (Query $query $value)
        (promote-placeholders (Query $pattern "$value") $placeholders)
     (let $values (unique-atom (collapse (match $db $query $value)))
       (collapse
         (let $candidate (superpose $values)
           (if (== $candidate ConditionalMissing)
               (empty)
               $candidate))))))

(= (conditional-case-table
      $db $key $row-pattern $placeholders $folds $target)
   (let (Query $pattern $observation)
        (promote-placeholders
          (Query $row-pattern (Observation "$fold" "$outcome" "$weight"))
          $placeholders)
     (let $observations (collapse (match $db $pattern $observation))
       (cons-atom
         (Cell $key AllFolds
           (conditional-observation-mass $observations AllFolds $target))
         (conditional-fold-cells $folds $observations $key $target)))))

(= (conditional-merge-tables ()) ())
(= (conditional-merge-tables $tables)
   (let ($head $tail) (decons-atom $tables)
     (append $head (conditional-merge-tables $tail))))

(= (conditional-cell () $_key $_fold) (Mass 0.0 0.0 0 0))
(= (conditional-cell $table $key $fold)
   (let ($head $tail) (decons-atom $table)
     (let (Cell $cell-key $cell-fold $mass) $head
       (if (and (== $key $cell-key) (== $fold $cell-fold))
           $mass
           (conditional-cell $tail $key $fold)))))

(= (conditional-rate $numerator $denominator $fallback)
   (if (> $denominator 0.0) (// $numerator $denominator) $fallback))

(= (conditional-stats
      $population-table $semantic-table $context-table $candidate-table
      $semantic-key $context-key $candidate-key $fold $fallback-rate)
   (let* (((Mass $total-positive $total-negative $_tp-count $_tn-count)
             (conditional-cell $population-table Population $fold))
          ($total (+ $total-positive $total-negative))
          ($base (conditional-rate $total-positive $total $fallback-rate))
          ((Mass $positive $negative $positive-count $negative-count)
             (conditional-cell $candidate-table $candidate-key $fold))
          ($support (+ $positive $negative))
          ($precision (conditional-rate $positive $support $base))
          ((Mass $semantic-positive $semantic-negative $_spc $_snc)
             (conditional-cell $semantic-table $semantic-key $fold))
          ((Mass $context-positive $context-negative $_cpc $_cnc)
             (conditional-cell $context-table $context-key $fold))
          ($semantic-precision
             (conditional-rate $semantic-positive
               (+ $semantic-positive $semantic-negative) $base))
          ($context-precision
             (conditional-rate $context-positive
               (+ $context-positive $context-negative) $base))
          ($parent (max $base (max $semantic-precision $context-precision)))
          ($coverage (conditional-rate $support $total 0.0))
          ($effect (- $precision $base))
          ($increment (- $precision $parent)))
     (Stats
       (Total $total-positive $total-negative $_tp-count $_tn-count)
       (Matched $positive $negative $positive-count $negative-count)
       (Rates $support $coverage $base $precision $effect $parent $increment
              (* $coverage $effect) (* $coverage $increment)))))

(= (conditional-fold-stat
      $population-table $semantic-table $context-table $candidate-table
      $semantic-key $context-key $candidate-key $fold $base
      $fold-min-support)
   (let $stats
        (conditional-stats
          $population-table $semantic-table $context-table $candidate-table
          $semantic-key $context-key $candidate-key $fold $base)
     (let (Stats $_total $_matched
                 (Rates $support $_coverage $_base $_precision $_effect
                        $_parent $_increment $_wracc $_incremental-wracc))
          $stats
       (Fold $fold $stats (>= $support $fold-min-support)))))

(= (conditional-usable-fold (Fold $fold $stats $usable))
   (if $usable (Fold $fold $stats $usable) (empty)))

(= (conditional-stability $usable)
   (Stable
     (min-atom
       (collapse
         (let (Fold $_f
                    (Stats $_t $_m
                      (Rates $_s $_c $_b $_p $effect $_pa $_i $_w $_iw))
                    True)
              (superpose $usable) $effect)))
     (min-atom
       (collapse
         (let (Fold $_f
                    (Stats $_t $_m
                      (Rates $_s $_c $_b $_p $_e $_pa $increment $_w $_iw))
                    True)
              (superpose $usable) $increment)))
     (min-atom
       (collapse
         (let (Fold $_f
                    (Stats $_t $_m
                      (Rates $_s $_c $_b $_p $_e $_pa $_i $_w $robust))
                    True)
              (superpose $usable) $robust)))
     (size-atom $usable)))

(= (conditional-stable-result
      $seed-ids $semantic $semantic-lineage $context $context-lineage
      $global $folds $min-folds $min-effect $min-increment)
   (let* (((Stats $_total $_matched
                  (Rates $_support $_coverage $_base $_precision $effect
                         $_parent $increment $wracc $incremental-wracc)) $global)
          ($usable (collapse
                     (conditional-usable-fold (superpose $folds))))
          ($usable-count (size-atom $usable)))
     (if (and (> $effect $min-effect)
              (and (> $increment $min-increment)
                   (>= $usable-count $min-folds)))
         (let (Stable $stable-effect $stable-increment $robust-wracc $_count)
              (conditional-stability $usable)
           (if (and (> $stable-effect $min-effect)
                    (> $stable-increment $min-increment))
               (RankedConditional
                 (- 0.0 $robust-wracc) (- 0.0 $incremental-wracc)
                 (- 0.0 $wracc) (- 0.0 $stable-increment)
                 (ConditionalRule
                   (Premises $semantic $context)
                   (Seed $seed-ids)
                   (Lineages $semantic-lineage $context-lineage)
                   $global
                   (Stable $stable-effect $stable-increment
                           $robust-wracc $usable-count)
                   (Folds $folds)))
               (empty)))
         (empty))))

(= (conditional-score
      $population-table $semantic-table $context-table
      $seeds $contexts $fold-values $candidate-cells
      $min-support $fold-min-support $min-folds $min-effect $min-increment)
   (let* (((Cell (Candidate $seed-key $functor $value)
                  AllFolds $_candidate-mass)
             (car-atom $candidate-cells))
          ($candidate (Candidate $seed-key $functor $value))
          ((Seed $seed-key $seed-ids $semantic $semantic-lineage)
             (superpose $seeds))
          ((Context $functor $context-lineage) (superpose $contexts))
          ($context (cons-atom $functor ("$x" $value)))
          ($semantic-key (Semantic $seed-key))
          ($context-key (Context $functor $value))
          ($global
            (conditional-stats
              $population-table $semantic-table $context-table $candidate-cells
              $semantic-key $context-key $candidate AllFolds 0.5)))
     (let (Stats $_total $_matched
                 (Rates $support $_coverage $base $_precision $_effect
                        $_parent $_increment $_wracc $_incremental-wracc))
          $global
       (if (>= $support $min-support)
           (let $folds
                (collapse
                  (conditional-fold-stat
                    $population-table $semantic-table $context-table
                    $candidate-cells $semantic-key $context-key $candidate
                    (superpose $fold-values) $base $fold-min-support))
             (conditional-stable-result
               $seed-ids $semantic $semantic-lineage $context $context-lineage
               $global $folds $min-folds $min-effect $min-increment))
           (empty)))))

(= (conditional-context-key
      (Cell (Context $functor $value) $_fold $_mass))
   (Context $functor $value))

(= (conditional-take $count $items)
   (if (or (<= $count 0) (== $items ()))
       ()
       (let ($head $tail) (decons-atom $items)
         (cons-atom $head (conditional-take (- $count 1) $tail)))))

(= (conditional-unrank $items)
   (if (== $items ())
       ()
       (let* ((($head $tail) (decons-atom $items))
              ((RankedConditional $_a $_b $_c $_d $rule) $head))
         (cons-atom $rule (conditional-unrank $tail)))))

(= (conditional-frequency-pattern-miner
      $population-tables $semantic-tables $context-tables $candidate-tables
      $seeds $contexts $fold-values
      $min-support $fold-min-support $min-folds
      $min-effect $min-increment $top-k $max-candidates)
   (let* (($population-table (conditional-merge-tables $population-tables))
          ($semantic-table (conditional-merge-tables $semantic-tables))
          ($context-table (conditional-merge-tables $context-tables))
          ($context-keys
            (unique-atom
              (collapse
                (conditional-context-key (superpose $context-table)))))
          ($candidate-count (size-atom $candidate-tables)))
     (if (> $candidate-count $max-candidates)
         (MiningLimit $candidate-count $max-candidates)
         (let* (($ranked
                   (collapse
                     (conditional-score
                       $population-table $semantic-table $context-table
                       $seeds $contexts $fold-values
                       (superpose $candidate-tables) $min-support
                       $fold-min-support $min-folds $min-effect $min-increment)))
                ($selected (conditional-take $top-k (sort-atom $ranked))))
           (ConditionalMining
             (Audit $candidate-count (size-atom $context-keys)
                    (size-atom $ranked) (size-atom $selected)
                    (conditional-cell
                      $population-table Population AllFolds))
             (conditional-unrank $selected))))))
