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
        -- in trace (show convolvedVals) (TMat convolvedVals : vs)
        in (TMat convolvedVals : vs)
        
instance AlgPrint ConvLayer where
    algPrint (ConvLayer filter bias prevOutput) = thisLayer ++ prevOutput
        where
            thisLayer = unlines ["conv layer: ", "filter: ", show filter, 
                                "bias: ", show bias, "--------------- \n"]


backwardConvLayer :: Filter -> Double -> BackPropNew -> (Filter, Double, Tensor)
-- again the head of the list is the output of the current layer
-- so the output of the conv. layer is ReLU(Z)
backwardConvLayer filter bias (BackPropCNN ((TMat reluZ) : (TMat x) : ts) (TMat dLdC)) = 
        -- ∂L/∂Z = ∂L/∂(Re(Z)) * ∂(Re(Z))/∂Z, ∂(Re(Z_mn))/∂Z_mn = 1 if Z_mn > 0, 0 otherwise
        -- keep in mind the * is pointwise multiplication because we did everything
        -- one index at a time then just made the matrix notation to keep it cleaner
    let dCdZ = mmap (\x -> if x > 0.0 then 1.0 else 0.0) reluZ
        dLdZ = dLdC ^**^ dCdZ -- matrix dot product
        -- ∂L/∂K = conv (x, ∂L/∂Z), can be derived
        delK = convolution x dLdZ 0.0
        -- ∂L/∂B = ∑ ∂L/∂Z
        delB = vfoldl' (+) 0.0 (matToVec dLdZ)
        -- icba writing it out but this is also derivable, assuming filter is square
        -- so we only have to pad with one integer information
        delX = convolution (padMatrixN (mheight filter - 1) dLdZ) (rotatePi filter) 0.0
    in (delK, delB, TMat delX)

learningRateConv = 0.004

instance (ConvLayer :<: f) => AlgBwdNew ConvLayer f where
    algBwdNew (ConvLayer filter bias nextLayersFunc) backPropForCurr = 
        let (delK, delB, delX) = backwardConvLayer filter bias backPropForCurr
            updFilter = filter -#- mmap (* learningRateConv) delK
            
            updBias = bias - (learningRateConv * delB) -- tis a scalar
            backPropForNext = BackPropCNN {cnnAs = tail (cnnAs backPropForCurr),
                                           cnnDs' = delX}
        in Op (inj (ConvLayer updFilter updBias (nextLayersFunc backPropForNext)))
-- NEED TO ACCOUNT FOR THE RELU!!!!! (i think i have?)