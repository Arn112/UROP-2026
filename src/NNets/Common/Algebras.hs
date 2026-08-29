-- Contains typeclasses for algebras used to eval the network structure. 
-- examples include for forward prop., back prop., and printing. 

{-
Each layer will have an instance of both to define how it should behave under forward and 
back propagation respectively. 

The gen functions are for the 'base case' of the eval, they correspond to after the input layer.
They are defined in another file (since they're not actually ever invoked, the bottom of the tree
ends up being the InputLayer).
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
-- optimisation point?

class AlgBwd g f where
    algBwd :: g (BackProp -> Free f a) -> (BackProp -> Free f a)
    -- g is the current layer backpropagated over 
    -- (the BackProp -> Free f a function is what will be stored in the 'hole' of the functor)
    -- f is the type of the final network

-- again, we require this for multi-type layered networks to typecheck, similar to algFwd
instance (AlgBwd g f, AlgBwd h f) => AlgBwd (g :+: h) f where
    algBwd (L fx) = algBwd fx
    algBwd (R gx) = algBwd gx

-- we create another algebra which allows us to print the network for debugging purposes
-- it makes sense that printing it recursively is the same process at heart
class Functor f => AlgPrint f where
    algPrint :: f String -> String
    -- only need the current layer's output for the rest of it

instance (AlgPrint f, AlgPrint g) => AlgPrint (f :+: g) where
    algPrint (L ft) = algPrint ft
    algPrint (R gt) = algPrint gt