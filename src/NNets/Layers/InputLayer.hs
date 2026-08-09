{-# LANGUAGE UndecidableInstances #-}
{-
Module: NNets.Types.Layers.InputLayer
Description: Implementation of the general input layer to be used in any network
-}

{-
This file contains the definition of the general input layer to be used by every
neural network. 
-}
module NNets.Layers.InputLayer where

import NNets.Common

data InputLayer a = InputLayer deriving Functor

-- smart constructor to inject into the free structure, depending on the type of the
-- final network (f):
inputLayer :: (InputLayer :<: f) => Free f a
inputLayer = Op (inj InputLayer)

-- the algFwd instance for this corresponds to initialising an empty list which
-- will take in the input to the NN and turn it into a list, so the rest of the 
-- activations can be appended in front.
instance AlgFwd InputLayer where
    algFwd InputLayer = (: []) -- simplified lambda

-- algBwd's job is to collapse a layer with a hole in it (for the backprop computation
-- so far) and incorporate the current layer's backprop rules into it. At the input layer,
-- we've got nothing left to propagate backwards, so we just return the InputLayer 
-- inserted into the correct type network. 
instance (InputLayer :<: f) => AlgBwd InputLayer f where
    algBwd InputLayer = const (Op (inj InputLayer))
    -- inj because we dk what f looks like, Op because needs to be Free f a


