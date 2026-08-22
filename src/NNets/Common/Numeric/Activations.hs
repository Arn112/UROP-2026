{-
Description: just for activation functions. Doesn't make sense to put them in vectors.
Will be more relevant when more are added
-}
module NNets.Common.Numeric.Activations where

import NNets.Common.Numeric.Vectors

-- IS THIS THE RIGHT FILE FOR THIS? PROBABLY NOT
learningRate :: Double = 0.4

-- sigmoid function. maps sigmoid over a vector
sigmoid :: Vector -> Vector
sigmoid = vmap (\x -> 1 / (1 + exp (-x)))

-- sigmoidInv' x = sigma' (sigma_inv(x)), i.e. derivative at inv. point.
-- (this is because input will usually be activation a, not raw z)
sigmoidInv' :: Vector -> Vector
sigmoidInv' = vmap (\x -> x * (1 - x))