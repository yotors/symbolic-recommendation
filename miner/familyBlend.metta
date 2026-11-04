;; Weights operate on existing family midranks, not on probabilities.
(= (family-blend $champion $structured $semantic $structured-weight)
   (+ $champion
      (* (- $structured-weight 0.5) (- $structured $semantic))))
