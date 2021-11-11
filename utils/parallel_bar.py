import random
import time

from joblib import Parallel, delayed
from tqdm import tqdm


all_bar_funcs = {'tqdm': lambda args: lambda x: tqdm(x, **args),
                 'False': lambda args: iter,
                 'None': lambda args: iter}


def ParallelExecutor(use_bar='tqdm', **joblib_args):

    def aprun(bar=use_bar, **tq_args):

        def tmp(op_iter):
            if str(bar) in all_bar_funcs.keys():
                bar_func = all_bar_funcs[str(bar)](tq_args)
            else:
                raise ValueError('Value %s not supported as bar type' % bar)
            return Parallel(**joblib_args)(bar_func(op_iter))

        return tmp

    return aprun