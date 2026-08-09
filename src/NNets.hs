{-
Module: NNets
Description: User facing API of all network layers and functions. 
-}
module NNets (trainMany, forwardProp, module NNets.Common, module NNets.Layers) where

import NNets.Common
import NNets.Layers
import Data.Array.Unboxed

genFwd :: a -> (Vector -> [Vector])
genFwd = const (: [])

-- I don't even think this gen case is ever attainable because there is
-- no pure values inside our tree
genBwd :: (InputLayer :<: f) => a -> (BackProp -> Free f a) 
genBwd _ = const (Op (inj InputLayer))

-- train :: (InputLayer :<: f, AlgFwd f, AlgBwd f f) => (Vector, Vector) -> Free f a -> Free f a
-- train (inp, desOut) nn = 
--     let algTrain = pairAlg algFwd algBwd
--         genTrain = pairGen genFwd genBwd
--         -- you don't need to run forwardPass for backwardPass to exist anymore
--         -- they are just functions composed independently, then chained
--         (forwardPass, backwardPass) = eval genTrain algTrain nn

--         h :: [Vector] -> BackProp
--         h vals = BackProp vals [] [] desOut

--     in (backwardPass . h . forwardPass) inp
forwardProp :: AlgFwd f => Free f a -> (Vector -> [Vector])
forwardProp = eval genFwd algFwd

backPropagate :: forall f a. (InputLayer :<: f, AlgBwd f f) => Free f a -> (BackProp -> Free f a)
backPropagate = eval genBwd algBwd

train :: (InputLayer :<: f, AlgFwd f, AlgBwd f f) => (Vector, Vector) -> Free f a -> Free f a
train (inp, desOut) nn = 
    let emptyArray = toVector [0]
        emptyTwoDimArray = toMatrix [[0]]
        -- just a placeholder, we never use the ws' or ds' on output layer

        h :: [Vector] -> BackProp
        h vals = BackProp vals emptyTwoDimArray emptyArray desOut 0

    in (backPropagate nn . h . forwardProp nn) inp

-- this function could be improved for performance potentially.
trainMany :: (InputLayer :<: f, AlgFwd f, AlgBwd f f) 
          => [(Vector, Vector)] -> Free f a -> Free f a
trainMany dataSet nn = foldr train nn dataSet

