module NNets.Debug.PrintNets (printNetwork) where

import NNets.Layers
import NNets.Common

-- There are no pure values in the tree so this isn't attainable either, because
-- fmap _ InputLayer = InputLayer (by deriving functor laws)
genPrint :: a -> String
genPrint = error "genPrint is invoked, but there are no pure values in the Free tree."

printNetwork :: AlgPrint f => Free f a -> IO ()
printNetwork nn = do
    putStrLn "NETWORK START: ----------------------------- "
    putStr $ showNetwork nn -- print doesn't work, put formats newlines right
    putStrLn "NETWORK END: ------------------------------- \n"  
    where
        showNetwork :: AlgPrint f => Free f a -> String
        showNetwork = eval genPrint algPrint

