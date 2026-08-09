{-
Module: NNets.Common.BackProp
Description: Contains the backprop type for storing backpropagation information.
-}
module NNets.Common.BackProp where

import NNets.Common.Numeric

data BackProp = BackProp {as :: [Vector], -- all (in/out)puts from forwardProp
                          ws' :: Weights, -- weights_l+1
                          ds' :: Deltas, -- delta_l+1
                          desiredOutput :: Vector, -- this layer new vals
                          layerIndex :: Int} -- output layer is index 0, it's weird
