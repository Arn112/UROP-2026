-- {-# LANGUAGE BangPatterns #-}
-- {-# LANGUAGE Strict #-}
{-
Module: NNets.Common.Free
Description: Contains the free monad definition and the eval and build functions
-}
module NNets.Common.Free where

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