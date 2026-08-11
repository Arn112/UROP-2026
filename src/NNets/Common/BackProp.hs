{-
Module: NNets.Common.BackProp
Description: Contains the backprop type for storing backpropagation information.
-}
module NNets.Common.BackProp where

import NNets.Common.Numeric

data BackProp = BackProp {as :: ![Vector], -- all (in/out)puts from forwardProp
                          ws' :: {-# UNPACK #-} !Weights, -- weights_l+1
                          ds' :: {-# UNPACK #-} !Deltas, -- delta_l+1
                          desiredOutput :: {-# UNPACK #-} !Vector, -- this layer new vals
                          layerIndex :: {-# UNPACK #-} !Int} -- output layer is index 0, it's weird
