-- Implementation of the general input layer to be used in any network

{-# LANGUAGE UndecidableInstances, ViewPatterns, PatternSynonyms #-}

module NNets.Layers.InputLayer where

import Data.List.NonEmpty (singleton)
import NNets.Common

data InputLayer k = InputLayer deriving Functor

-- smart constructor to inject into the free structure, depending on the type of the
-- final network (f):
inputLayer :: (InputLayer :<: f) => Free f a
inputLayer = Op (inj InputLayer)

pattern InputLayer' <- (prj -> Just InputLayer)

-- the algFwd instance for this corresponds to initialising an empty list which
-- will take in the input to the NN and turn it into a list, so the rest of the 
-- activations can be appended in front. Notice this lambda is equivalent to the 
-- 'singleton' function.
instance AlgFwd InputLayer where
    algFwd InputLayer = singleton

-- algBwd's job is to collapse a layer with a hole in it (for the backprop computation
-- so far) and incorporate the current layer's backprop rules into it. At the input layer,
-- we've got nothing left to propagate backwards, so we just return the InputLayer 
-- inserted into the correct type network. 
instance (InputLayer :<: f) => AlgBwd InputLayer f where
    algBwd InputLayer = const (Op (inj InputLayer))
    -- inj because we don't know what f looks like, Op because needs to be Free f a

instance AlgPrint InputLayer where
    algPrint InputLayer = "input layer \n"



