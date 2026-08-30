{-# LANGUAGE UndecidableInstances, ViewPatterns, PatternSynonyms #-}

module NNets.Layers.PoolLayer where

import NNets.Common
import Debug.Trace (trace)

data PoolLayer a = PoolLayer SpatExt Stride a deriving Functor

poolLayer :: (PoolLayer :<: f) => SpatExt -> Stride -> Free f ()
poolLayer spatialExtent stride = Op (inj $ PoolLayer spatialExtent stride (Pure ()))

pattern PoolLayer' st str nn' <- (prj -> Just (PoolLayer st str nn'))

instance AlgFwdNew PoolLayer where
    algFwdNew (PoolLayer spatialExtent stride forwardPass) inp =
        let vs@(TMat vals : _) = forwardPass inp
            pooledVals = pool vals spatialExtent stride
        in trace (show pooledVals) (TMat pooledVals : vs)

instance AlgPrint PoolLayer where
    algPrint (PoolLayer spatext stride prevOutput) = thisLayer ++ prevOutput
        where
            thisLayer = unlines ["pool layer: ", "spatial extent: " ++ show spatext, 
                                "stride: " ++ show stride, "--------------- \n"]
