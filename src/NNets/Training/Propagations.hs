-- definition of forward and backward propagation functions. required to define training functions. 

module NNets.Training.Propagations where

import NNets.Common
import NNets.Layers
import Data.Array.Unboxed
import Data.List.NonEmpty (NonEmpty((:|)), singleton)

-- There are no pure values inside the tree, so this is never reachable.
genFwd :: a -> (Vector -> NonEmpty Vector)
genFwd = error "genFwd is invoked, but there are no pure values in the Free tree."

-- There are no pure values inside our tree, so this is never reachable.
genBwd :: (InputLayer :<: f) => a -> (BackProp -> Free f a) 
genBwd = error "genBwd is invoked, but there are no pure values in the free tree"

forwardProp :: AlgFwd f => Free f a -> (Vector -> NonEmpty Vector)
forwardProp = eval genFwd algFwd
-- How much work does the call to eval do? Does it build the full function?
-- or does it build it and the result at the same time only when it's required / applied to 
-- the network? Will need to analyse a little more. 

backPropagate :: forall f a. (InputLayer :<: f, AlgBwd f f) => Free f a -> (BackProp -> Free f a)
backPropagate = eval genBwd algBwd
-- Similar to the above for this - how much is actually being done?