{-# LANGUAGE UndecidableInstances, ViewPatterns, PatternSynonyms #-}

module NNets.Layers.ConvLayer where

import NNets.Common
import Debug.Trace (trace)

data ConvLayer a = ConvLayer Filter Double a deriving Functor

convLayer :: (ConvLayer :<: f) => Filter -> Double -> Free f ()
convLayer filter biases = Op (inj $ ConvLayer filter biases (Pure ()))

pattern ConvLayer' filter biases nn' <- (prj -> Just (ConvLayer filter biases nn'))

instance AlgFwdNew ConvLayer where
    algFwdNew (ConvLayer filter bias forwardPass) inp =
        let vs@(TMat vals : _) = forwardPass inp
            convolvedVals = convolution vals filter bias
        in trace (show convolvedVals) (TMat convolvedVals : vs)
        
instance AlgPrint ConvLayer where
    algPrint (ConvLayer filter bias prevOutput) = thisLayer ++ prevOutput
        where
            thisLayer = unlines ["conv layer: ", "filter: ", show filter, 
                                "bias: ", show bias, "--------------- \n"]