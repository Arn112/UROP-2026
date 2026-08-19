{-# LANGUAGE UndecidableInstances, ViewPatterns, PatternSynonyms #-}
{-
Module: NNets.Types.Layers.InputLayer
Description: Implementation of the general input layer to be used in any network
-}

{-
This file contains the definition of the general dense layer to be used by every
neural network. 
-}
module NNets.Layers.DenseLayer ( denseLayer, DenseLayer, pattern DenseLayer' ) where

import NNets.Common
import Data.Array.Unboxed
import Data.List.NonEmpty (NonEmpty((:|)))
import Debug.Trace

data DenseLayer k = DenseLayer {-# UNPACK #-} !Weights {-# UNPACK #-} !Biases k deriving Functor

-- smart constructor:
denseLayer :: (DenseLayer :<: f) => Weights -> Biases -> Free f ()
denseLayer ws bs = Op (inj $ DenseLayer ws bs (Pure ()))

-- smart destructor? Idk what the right name is. Use case is for printing
-- etc. where you don't know the shape of the network type but need to pattern match
-- on the different layers. 
pattern DenseLayer' ws bs nn' <- (prj -> Just (DenseLayer ws bs nn'))

-- eval on a free structure recursively evaluates the hole of the structure, 
-- then the current layer. Here this corresponds to computing the function which
-- forward propagates through the network up to this layer, then computing forward
-- prop through this layer and appending the results to the front.
instance AlgFwd DenseLayer where
    algFwd (DenseLayer wl bl forwardPass) inp = 
        let vals :| vs = forwardPass inp
            !cvals = sigmoid ((wl #> vals) ^+^ bl)
        in cvals :| (vals : vs)

-- helper function for computing backpropagation through a DenseLayer. 
-- takes in the weights & biases of the layer, and the current backprop info
-- being fed in, and returns the updated weights and biases of the layer. 
{-# INLINE backward #-} -- hopefully it gets rid of the tuple 
backward :: Weights -> Biases -> BackProp -> (Weights, Biases, BackProp)
backward ws bs (BackProp al (alPrev : als) ws' ds' desiredOutput layerIndex) =
    let !dlNew = case layerIndex of
            0 -> (al ^-^ desiredOutput) ^*^ sigmoidInv' al
            -- delta_j(L) = (a_j(L) - y_j) * sigma'(z_j(L))
            _ -> (transposeA ws' #> ds') ^*^ sigmoidInv' al
            -- delta_l = ((w_l+1)^T delta_l+1) ^*^ sigma'(z_l)
        !wlNew = ws -#- mmap (*learningRate) (dlNew >< alPrev)
        -- ∂C/∂w_jk = a_k(l-1) * delta_j(l)
        !blNew = bs ^-^ vmap (*learningRate) dlNew
        -- ∂C/∂b_j = delta_j(l)
    in (wlNew, blNew, BackProp alPrev als ws dlNew desiredOutput (layerIndex + 1))


-- eval will give us the function for backpropagating over the rest of
-- the deeper layers (I.e. towards the input layer). We just need to 
-- incorporate the current layer's changes into it, and insert the updated layer
-- to build up the updated network. 
instance (DenseLayer :<: f) => AlgBwd DenseLayer f where
    algBwd (DenseLayer ws bs nextLayersFunc) backPropForCurr = 
        let (!wlNew, !blNew, !backPropForNext) = backward ws bs backPropForCurr
        in Op (inj (DenseLayer wlNew blNew (nextLayersFunc backPropForNext)))

-- this is fine because we right associate.
instance AlgPrint DenseLayer where
    algPrint (DenseLayer ws bs prevOutput) = thisLayer ++ prevOutput
        where
            thisLayer = unlines ["dense layer: ", "ws: ", show ws, "bs: ", 
                                show bs, "--------------- \n"]
