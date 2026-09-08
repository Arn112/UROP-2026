{-
Module: NNets.Common.BackProp
Description: Contains the backprop type for storing backpropagation information.
-}
module NNets.Common.BackProp where

import NNets.Common.Numeric

data BackProp = BackProp {a :: !Vector, -- first (in/out)put from forwardProp
                          as :: ![Vector], -- rest (in/out)puts from forwardProp
                          ws' :: {-# UNPACK #-} !Weights, -- weights_l+1
                          ds' :: {-# UNPACK #-} !Deltas, -- delta_l+1
                          desiredOutput :: {-# UNPACK #-} !Vector, -- output layer new vals
                          layerIndex :: {-# UNPACK #-} !Int} -- output layer is index 0, it's weird

-------------- CNN Stuff ------------------
{-
There has to be a cleaner way to write BackProp? The shared features are deltas and activations, 
but denselayers require ws+l+1 and bs_l+1 whereas convs only require delX_l+1
-} 
-- data BackPropNew = BackPropMLP {mlpAs :: [Tensor], mlpWs' :: Weights, mlpDs' :: Deltas, desOut :: Vector}
--                 | BackPropCNN {cnnAs :: [Tensor], cnnDs' :: Tensor} -- activations and ∂L/∂Z, Z input to next layer.

data BackPropNew = BackPropMLP {mlpA :: !Tensor, 
                                mlpAs :: ![Tensor], 
                                mlpWs' :: {-# UNPACK #-} !Weights, 
                                mlpDs' :: {-# UNPACK #-} !Deltas, 
                                desOut :: {-# UNPACK #-} !Vector,
                                layerIndex' :: {-# UNPACK #-} !Int}
                | BackPropCNN {cnnAs :: [Tensor], cnnDs' :: Tensor} -- activations and ∂L/∂Z, Z input to next layer.

