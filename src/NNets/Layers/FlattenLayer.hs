{-# LANGUAGE UndecidableInstances, ViewPatterns, PatternSynonyms #-}

module NNets.Layers.FlattenLayer where

import NNets.Common
import Debug.Trace (trace)

-- the need for a flatten layer will be seen when describing the algebra
-- we have two parameters for height and width, corresponding to the result
-- of the previous pooling / conv layer. this will need to be given in 
-- manually for now. 
data FlattenLayer a = FlattenLayer Int Int a deriving Functor 

flattenLayer :: (FlattenLayer :<: f) => Int -> Int -> Free f ()
flattenLayer height width = Op (inj $ FlattenLayer height width (Pure ()))

pattern FlattenLayer' h w nn <- (prj -> Just (FlattenLayer h w nn))

instance AlgFwdNew FlattenLayer where
    algFwdNew (FlattenLayer height width forwardPass) inp =
        let vs@(TMat poolLayerOutput : _) = forwardPass inp
        -- in trace (show $ matToVec poolLayerOutput) (TVec (matToVec poolLayerOutput) : vs)
        in (TVec (matToVec poolLayerOutput) : vs)
    -- we turn the 2D output of the prev pool (or conv) layer into a vector
    -- then add it to the activations list so the next denselayer
    -- sees the right thing

instance AlgPrint FlattenLayer where
    algPrint (FlattenLayer height width prevOutput) = thisLayer ++ prevOutput
        where
            thisLayer = unlines ["flatten layer: ", "height: " ++ show height,
                                "width: " ++ show width, "--------------- \n"]

instance (FlattenLayer :<: f) => AlgBwdNew FlattenLayer f where
    algBwdNew (FlattenLayer height width nextLayersFunc) backPropForCurr = 
        let TVec al = head $ mlpAs backPropForCurr 
            flattened = (transposeA (mlpWs' backPropForCurr) #> mlpDs' backPropForCurr)
            reshapedMat = reshape height width flattened -- reshape the vector of errors into a matrix
            -- ignore the element we added to the AS during forward prop - repeated result:
            backPropForNext = BackPropCNN {cnnDs' = TMat reshapedMat, cnnAs = mlpAs backPropForCurr}
            -- we DON'T do tail of mlpAs because we refactored it so the head is a different element altogether
        in Op (inj (FlattenLayer height width (nextLayersFunc backPropForNext)))
-- ok this needs a bit more explanation than I initially thought. 
-- The thing we need to reshape is actually the error of the INPUT layer, but
-- previous I was calculating the error of the layer after that. So you just do
-- the same thing as the MLP backprop for the dlNew case, but without the 
-- sigmoid because there was no calculation even done in this layer hence
-- no activation function was used. 