-- {-# LANGUAGE UnboxedTuples, BangPatterns #-}
-- {-# LANGUAGE Strict #-}
{-
Module: NNets
Description: User facing API of all network layers and functions. 
-}
module NNets (trainMany, forwardProp, module NNets.Common, module NNets.Layers) where

import NNets.Common
import NNets.Layers
import Data.Array.Unboxed
import Data.List.NonEmpty (NonEmpty((:|)), singleton)

genFwd :: a -> (Vector -> NonEmpty Vector)
genFwd = const singleton

-- I don't even think this gen case is ever attainable because there is
-- no pure values inside our tree
genBwd :: (InputLayer :<: f) => a -> (BackProp -> Free f a) 
genBwd _ = const (Op (inj InputLayer))

forwardProp :: AlgFwd f => Free f a -> (Vector -> NonEmpty Vector)
forwardProp = eval genFwd algFwd
-- I'm not sure why eval works fast in this case. I would assume that building the function
-- creates huge thunks, but it doesn't seem to show much in the benchmark. Maybe it's because
-- we only do forwardProp nn on the input, so then the laziness helps it perhaps avoid building
-- the lambda as it threads through the outputs of each step?

backPropagate :: forall f a. (InputLayer :<: f, AlgBwd f f) => Free f a -> (BackProp -> Free f a)
backPropagate = eval genBwd algBwd

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
trainMany dataSet nn = foldr train nn dataSet
-- trainMany dataSet nn = foldl' (flip train) nn dataSet
-- trainMany dataSet nn = go train dataSet nn 
--     where
--         go _ [] !nn = nn
--         go train (d:ds) !nn = go train ds (train d nn)


