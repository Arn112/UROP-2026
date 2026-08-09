{-
Defines some common type aliases used throughout the library
-}
module NNets.Common.Numeric.Types where

import NNets.Common.Numeric.Vectors

type Weights = Matrix
type Biases = Vector
type Deltas = Vector

-- DECIDE WHICH FILE THIS NEEDS TO GO IN
learningRate :: Double = 0.4