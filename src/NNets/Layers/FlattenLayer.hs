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
        in trace (show $ matToVec poolLayerOutput) (TVec (matToVec poolLayerOutput) : vs)
    -- we turn the 2D output of the prev pool (or conv) layer into a vector
    -- then add it to the activations list so the next denselayer
    -- sees the right thing

instance AlgPrint FlattenLayer where
    algPrint (FlattenLayer height width prevOutput) = thisLayer ++ prevOutput
        where
            thisLayer = unlines ["flatten layer: ", "height: " ++ show height,
                                "width: " ++ show width, "--------------- \n"]
