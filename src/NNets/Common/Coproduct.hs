-- Contains functor coproduct definition and definition of a 'contains' style operator
-- following data types a la carte, as the original paper does.

{-# LANGUAGE TypeOperators #-}

module NNets.Common.Coproduct where

infixr 5 :+:
data (f :+: g) a = L (f a) | R (g a) deriving Functor

class (Functor sub, Functor sup) => sub :<: sup where
    inj :: sub a -> sup a 
    prj :: sup a -> Maybe (sub a) -- need not be a bijection

instance Functor f => f :<: f where
    inj = id

    prj = Just

instance (Functor f, Functor (f :+: g)) => f :<: (f :+: g) where
    inj = L

    prj (L x) = Just x
    prj _ = Nothing

instance (Functor g, Functor (f :+: g)) => g :<: (f :+: g) where
    inj = R

    prj (R x) = Just x
    prj _ = Nothing

instance {-# OVERLAPPABLE #-} 
        (Functor f, Functor g, Functor h, f :<: g) => f :<: (h :+: g) where
    inj = R . inj
    -- this requires right associativity since we can only ever recurse 
    -- on the RHS

    prj (R x) = prj x
    prj _ = Nothing
