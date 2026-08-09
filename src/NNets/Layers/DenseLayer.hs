{-# LANGUAGE UndecidableInstances #-}
{-
Module: NNets.Types.Layers.InputLayer
Description: Implementation of the general input layer to be used in any network
-}

{-
This file contains the definition of the general dense layer to be used by every
neural network. 
-}
module NNets.Layers.DenseLayer ( denseLayer, DenseLayer ) where

import NNets.Common
import Data.Array.Unboxed

data DenseLayer a = DenseLayer Weights Biases a deriving Functor

-- smart constructor:
denseLayer :: (DenseLayer :<: f) => Weights -> Biases -> Free f ()
denseLayer ws bs = Op (inj $ DenseLayer ws bs (Pure ()))

-- eval on a free structure recursively evaluates the hole of the structure, 
-- then the current layer. Here this corresponds to computing the function which
-- forward propagates through the network up to this layer, then computing forward
-- prop through this layer and appending the results to the front.
instance AlgFwd DenseLayer where
    algFwd (DenseLayer wl bl forwardPass) = (\(vals : vs) -> 
        let cvals = sigmoid ((wl #> vals) ^+^ bl) in (cvals : vals : vs)) 
        . forwardPass

-- helper function for computing backpropagation through a DenseLayer. 
-- takes in the weights & biases of the layer, and the current backprop info
-- being fed in, and returns the updated weights and biases of the layer. 
backward :: Weights -> Biases -> BackProp -> (Weights, Biases, Deltas)
backward ws bs (BackProp (al : alPrev : as) ws' ds' desiredOutput layerIndex) =
    let dlNew = case layerIndex of
            0 -> (al ^-^ desiredOutput) ^*^ sigmoidInv' al
            -- delta_j(L) = (a_j(L) - y_j) * sigma'(z_j(L))
            _  -> (transposeA ws' #> ds')  ^*^ sigmoidInv' al
            -- delta_l = ((w_l+1)^T delta_l+1) ^*^ sigma'(z_l)
        wlNew = ws -#- amap (*learningRate) (dlNew >< alPrev)
        -- ∂C/∂w_jk = a_k(l-1) * delta_j(l)
        blNew = bs ^-^ amap (*learningRate) dlNew
        -- ∂C/∂b_j = delta_j(l)
    in (wlNew, blNew, dlNew)

-- eval will give us the function for backpropagating over the rest of
-- the deeper layers (I.e. towards the input layer). We just need to 
-- incorporate the current layer's changes into it, and insert the updated layer
-- to build up the updated network. 
instance (DenseLayer :<: f) => AlgBwd DenseLayer f where
    algBwd (DenseLayer ws bs nextLayersFunc) backPropForCurr = 
        let (wlNew, blNew, dlNew) = backward ws bs backPropForCurr
            backPropForNext = BackProp {ws' = ws, -- next layer's weights now this layer's
                                ds' = dlNew, -- same for next layer's errors
                                as = tail (as backPropForCurr), -- same for outputs
                                desiredOutput = desiredOutput backPropForCurr, -- same desired output
                                layerIndex = layerIndex backPropForCurr + 1} -- increment index

        in Op (inj (DenseLayer wlNew blNew (nextLayersFunc backPropForNext)))
