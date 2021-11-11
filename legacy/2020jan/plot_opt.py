import ax
from ax.plot.slice import plot_slice
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.plot.contour import plot_contour
from ax import load
import argparse

from ax import ParameterType
from ax import RangeParameter
from ax import SearchSpace
from ax import SimpleExperiment
from ax import save
from ax import load
from ax.modelbridge import get_sobol
from ax.modelbridge.factory import get_botorch


from config import Config
if __name__ == "__main__":
    config = Config()
    exp = load(config.BO_expe_path + '.json')
    exp.evaluation_function = run
    del exp.trials[19]
    gpei = ax.Models.GPEI(experiment=exp, data=exp.eval())
    cplot = plot_contour(model = gpei, param_x = 'do', param_y = 'l2', metric_name='objective',lower_is_better=True)
    

