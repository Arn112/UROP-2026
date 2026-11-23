-- defines and exposes the training functions and parameters required for the user.

module NNets.Train (
    trainMany,
    forwardProp,
    learningRate
) where

import NNets.Training.Propagations
import NNets.Common
import NNets.Layers
import Data.Array.Unboxed
import Data.List.NonEmpty (NonEmpty((:|)), singleton)

train :: (InputLayer :<: f, AlgFwd f, AlgBwd f f) => (Vector, Vector) -> Free f a -> Free f a
train (inp, desOut) nn = 
    let emptyArray = toVector [0]
        emptyTwoDimArray = toMatrix [[0]]
        -- just a placeholder, we never use the ws' or ds' on output layer

        h :: NonEmpty Vector -> BackProp
        h (val :| vals) = BackProp val vals emptyTwoDimArray emptyArray desOut 0

    in (backPropagate nn . h . forwardProp nn) inp

-- Inlining this function and using the inlined style definition (with a helper
-- which captures the recursion instead) causes a large spike at the end, so
-- sticking with foldl', but it's not been tested extensively.
trainMany :: (InputLayer :<: f, AlgFwd f, AlgBwd f f) 
          => [(Vector, Vector)] -> Free f a -> Free f a
trainMany dataSet nn = foldl' (flip train) nn dataSet