import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

from src import preprocess, training, diffe
import importlib

importlib.reload(preprocess)
importlib.reload(training)
importlib.reload(diffe)

preprocess.main(channels = ['Fp1', 'Fp2'])