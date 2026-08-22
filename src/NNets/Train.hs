-- defines and exposes the training functions required for the user

module NNets.Train (
    trainMany,
    forwardProp
) where

import NNets.Training.Propagations
import NNets.Common
import NNets.Layers
-- import NNets.PrintNets
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

-- {-# INLINE trainMany #-} this causes a large spike at the end,
-- so removing it but not tested extensively.
trainMany :: (InputLayer :<: f, AlgFwd f, AlgBwd f f) 
          => [(Vector, Vector)] -> Free f a -> Free f a
-- trainMany dataSet nn = foldr train nn dataSet
trainMany dataSet nn = foldl' (flip train) nn dataSet
-- trainMany dataSet nn = go train dataSet nn 
--     where
--         go _ [] !nn = nn
--         go train (d:ds) !nn = go train ds (train d nn)