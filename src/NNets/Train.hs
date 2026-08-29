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

trainMany :: (InputLayer :<: f, AlgFwd f, AlgBwd f f) 
          => [(Vector, Vector)] -> Free f a -> Free f a
trainMany dataSet nn = foldl' (flip train) nn dataSet
-- There might be an advantage to inlining and testing with the below definition instead, 
-- as in Jamie's benchmarking lecture. Maybe try it later.
-- trainMany dataSet nn = go train dataSet nn 
--     where
--         go _ [] !nn = nn
--         go train (d:ds) !nn = go train ds (train d nn)