{-
Module: NNets.Common.Algebras
Description: Contains typeclasses for forward and backward algebras.
-}

{-
Each layer will have an instance of both to define how it should behave under forward and 
back propagation respectively. 

The gen functions are for the 'base case' of the eval, they correspond to after the input layer
-}
module NNets.Common.Algebras where

import Data.List.NonEmpty (NonEmpty)

import NNets.Common.BackProp
import NNets.Common.Numeric
import NNets.Common.Coproduct
import NNets.Common.Free

class Functor f => AlgFwd f where
    algFwd :: f (Vector -> NonEmpty Vector) -> (Vector -> NonEmpty Vector)

-- we require this for more than one type of layered network:
instance (AlgFwd f, AlgFwd g) => AlgFwd (f :+: g) where
    algFwd (L ft) = algFwd ft
    algFwd (R gt) = algFwd gt
-- if f is large, then this rolls out to a bunch of Ls,Rs. Could be a potential
-- optimisation

class AlgBwd g f where
    algBwd :: g (BackProp -> Free f a) -> (BackProp -> Free f a)
    -- g is the current layer backpropagated over 
    -- (the BackProp -> Free f a function is what will be stored in the 'hole' of the functor)
    -- f is the type of the final network

-- again, we require this for multi-type layered networks to typecheck, similar to algFwd
instance (AlgBwd g f, AlgBwd h f) => AlgBwd (g :+: h) f where
    algBwd (L fx) = algBwd fx
    algBwd (R gx) = algBwd gx