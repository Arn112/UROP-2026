-- Defines some common type aliases used throughout the library
-- More will be added as CNNs and RNNs are added (eventually)

module NNets.Common.Numeric.Types where

import NNets.Common.Numeric.Vectors

type Weights = Matrix
type Biases = Vector
type Deltas = Vector