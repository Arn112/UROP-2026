-- defines and exposes the training functions and parameters
-- required for the user.

module NNets.Train (
    trainMany, trainManyCNN,
    forwardProp,
    forwardPropConv,
    learningRate,
    module NNets.Training.Initialise,
    trainCNN
) where

import NNets.Training.Propagations
import NNets.Training.Initialise
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



------------------ New instances for CNN integration: --------------------
-- unforunately we can only have desired output as a Vector right now. To change this, we 
-- would have to add a layerIndex == 0 condition for every single layer and describe what to do
-- if that was the final layer. It isn't too bad but I haven't added it yet. 
trainCNN :: (InputLayer :<: f, AlgFwdNew f, AlgBwdNew f f) => (Tensor, Vector) -> Free f a -> Free f a
trainCNN (inp, desout) nn =
    let emptyArray = toVector [0]
        emptyTwoDimArray = toMatrix [[0]]

        h :: [Tensor] -> BackPropNew
        h (val : vals) = BackPropMLP val vals emptyTwoDimArray emptyArray desout 0
    in (backPropagateCNN nn . h . forwardPropConv nn) inp

trainManyCNN :: (InputLayer :<: f, AlgFwdNew f, AlgBwdNew f f) => [(Tensor, Vector)] -> Free f a -> Free f a
trainManyCNN dataset nn = foldl' (flip trainCNN) nn dataset