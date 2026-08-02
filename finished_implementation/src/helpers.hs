{-# LANGUAGE TypeOperators #-}
module Helpers where

------------------------------- Free Monad ------------------------------------
data Free f a = Op (f (Free f a)) | Pure a deriving Functor

instance Functor f => Applicative (Free f) where
    pure = Pure

    (<*>) :: Free f (a -> b) -> Free f a -> Free f b
    Pure f <*> y = fmap f y
    Op f <*> y = Op (fmap (<*> y) f) 

instance Functor f => Monad (Free f) where
    return = pure
    Pure x >>= f = f x
    Op k >>= f = Op (fmap (>>= f) k)

-- Free monad version of cata:
eval :: Functor f => (a -> b) -> (f b -> b) -> Free f a -> b
eval gen alg (Pure x) = gen x
eval gen alg (Op t) = (alg . fmap (eval gen alg)) t

-- Free monad version of ana:
build :: Functor f => (b -> f b) -> b -> Free f a
build coalg seed = Op (fmap (build coalg) (coalg seed))

------------------------------- Coproducts ------------------------------------
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


-- I'm not sure if we need the following instance but:
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


