-- Contains the backprop type for storing backpropagation information.

module NNets.Common.BackProp where

import NNets.Common.Numeric

data BackProp = BackProp {a :: !Vector, -- first (in/out)put from forwardProp
                          as :: ![Vector], -- rest (in/out)puts from forwardProp
                          ws' :: {-# UNPACK #-} !Weights, -- weights_l+1
                          ds' :: {-# UNPACK #-} !Deltas, -- delta_l+1
                          desiredOutput :: {-# UNPACK #-} !Vector, -- this layer new vals
                          layerIndex :: {-# UNPACK #-} !Int} -- output layer is index 0, it's weird
