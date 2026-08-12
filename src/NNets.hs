{-# LANGUAGE UnboxedTuples, BangPatterns #-}
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

backPropagate :: forall f a. (InputLayer :<: f, AlgBwd f f) => Free f a -> (BackProp -> Free f a)
backPropagate = eval genBwd algBwd

-- banana split property of folds: any pair of folds can be combined to a 
-- single fold which generates a pair:
-- pairGen :: (a -> b) -> (a -> c) -> a -> (b, c)
-- pairGen f g x = (f x, g x)

-- pairAlg :: Functor f => (f b -> b) -> (f c -> c) -> f (b, c) -> (b, c)
-- pairAlg algB algC fbc = (algB (fmap fst fbc), algC (fmap snd fbc))

-- train :: (InputLayer :<: f, AlgFwd f, AlgBwd f f) => (Vector, Vector) -> Free f a -> Free f a
-- train (inp, desOut) nn = 
--     let algTrain = pairAlg algFwd algBwd
--         genTrain = pairGen genFwd genBwd
--         emptyArray = toVector [0]
--         emptyTwoDimArray = toMatrix [[0]]
--         -- you don't need to run forwardPass for backwardPass to exist anymore
--         -- they are just functions composed independently, then chained
--         (forwardPass, backwardPass) = eval genTrain algTrain nn
--         h :: [Vector] -> BackProp
--         h vals = BackProp vals emptyTwoDimArray emptyArray desOut 0
--     in (backwardPass . h . forwardPass) inp

-- data Passes a b = Passes {fwd :: !a, bwd :: !b}

-- pairGenP :: (a -> b) -> (a -> c) -> a -> Passes b c
-- pairGenP f g x = Passes {fwd = f x, bwd = g x}

-- pairAlgP :: Functor f => (f b -> b) -> (f c -> c) -> f (Passes b c) -> Passes b c
-- pairAlgP algB algC fbc = Passes {fwd = algB (fmap fwd fbc), bwd = algC (fmap bwd fbc)}

-- train :: (InputLayer :<: f, AlgFwd f, AlgBwd f f) => (Vector, Vector) -> Free f a -> Free f a
-- train (inp, desOut) nn = 
--     let algTrainP = pairAlgP algFwd algBwd
--         genTrainP = pairGenP genFwd genBwd
--         emptyArray = toVector [0]
--         emptyTwoDimArray = toMatrix [[0]]
        
--         !(Passes fwdPass bwdPass) = eval genTrainP algTrainP nn
        
--         h :: [Vector] -> BackProp
--         h vals = BackProp vals emptyTwoDimArray emptyArray desOut 0

--     in (bwdPass . h . fwdPass) inp

train :: (InputLayer :<: f, AlgFwd f, AlgBwd f f) => (Vector, Vector) -> Free f a -> Free f a
train (inp, desOut) nn = 
    let emptyArray = toVector [0]
        emptyTwoDimArray = toMatrix [[0]]
        -- just a placeholder, we never use the ws' or ds' on output layer

        h :: NonEmpty Vector -> BackProp
        h (val :| vals) = BackProp val vals emptyTwoDimArray emptyArray desOut 0

    in (backPropagate nn . h . forwardProp nn) inp

-- this function could be improved for performance potentially.
trainMany :: (InputLayer :<: f, AlgFwd f, AlgBwd f f) 
          => [(Vector, Vector)] -> Free f a -> Free f a
trainMany dataSet nn = foldr train nn dataSet

