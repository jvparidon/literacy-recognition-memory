# -*- coding: utf-8 -*-
# jvparidon@gmail.com
import pymc3 as pm
import pandas as pd
import numpy as np
import pickle
import os
import copy
import scipy.stats
from scipy.special import logit
from scipy.special import expit as invlogit
import seaborn as sns
import matplotlib.pyplot as plt


def d_to_logodds(d):
    return np.array(d) * (np.pi / np.sqrt(3))


def logodds_to_d(logodds):
    return np.array(logodds) * (np.sqrt(3) / np.pi)


def standardize(df):
    """Standardize using standard method

    Centers variables and standardizes them by dividing by the standard deviation
    (as opposed to dividing by double the standard deviation) useful for outcome variables, for instance

    :param df:
    :return: standardized dataframe columns
    """
    return (df - df.mean()) / df.std()


def standardize_gelman(df):
    """Standardize cf. Gelman recommendation (to get scale comparable to unstandardized binary predictors)

    Centers variables and standardizes them by dividing by double the standard deviation
    (as opposed to dividing by just the standard deviation) to put continuous variables on the same scale
    as binary variables (to aid with interpreting interactions conditioned on binary variables)

    :param df:
    :return: standardized dataframe columns
    """
    return (df - df.mean()) / (2 * df.std())


def expand_binomial(df, success_col, n):
    # repeat each row for success and n - success times
    df_success = df.reindex(df.index.repeat(df[success_col]))
    df_success[success_col + '_bernoulli'] = 1
    df_failure = df.reindex(df.index.repeat(n - df[success_col]))
    df_failure[success_col + '_bernoulli'] = 0
    df = pd.concat([df_success, df_failure]).reset_index()
    return df


class ModelPickler():
    """Object for pickling and unpickling BAMBI or PyMC3 models.

    Stores models in a dict and serializes or unserializes that dict as needed.

    :param fname: Filename for pickle file serialized models will be written to.
    :param models: Dictionary that contains models, indexed by model name.
    """
    def __init__(self, fname):
        """Constructor method
        """
        self.fname = fname
        self.models = dict()
        if os.path.exists(self.fname):
            self.load()

    def add(self, model, label):
        """Method for adding models to the ModelPickler.

        Adds models to the dict that hold models, then serializes whole dict
        and writes it to the pickle file.

        :param model: BAMBI or PyMC3 model to add to ModelPickler.
        :param label: Model name, used as dict key.
        """
        self.models[label] = model
        with open(self.fname, 'wb') as f:
            pickle.dump(self.models, f)

    def load(self):
        """Method for loading models from pickle file.

        Unserializes models dict from pickle file.
        Filename is taken from object initialization.
        """
        with open(self.fname, 'rb') as f:
            self.models.update(pickle.load(f))


def forestplot(model, bambi=False, transform=np.array, vline_label=None, rhat=False, **kwargs):
    """Modified forestplot function

    Forestplot function from PyMC3, adapted to automatically plot only relevant effects
    for a BAMBI or PyMC3 model and to add a vertical no effect line to aid with interpreting coefficients

    :param trace: BAMBI or PyMC3 model object
    :param transform: function to transform trace (pass np.exp for logistic regression)
    :param kwargs: keyword args for PyMC3 forestplot function
    :returns: matplotlib subplot object with forestplot for trace
    """
    if bambi:
        trace = model.backend.trace
        varnames = sorted(model.fixed_terms.keys())
    else:
        trace = model
        varnames = sorted(trace.varnames)
    pm.forestplot(trace,
                  varnames=varnames,
                  transform=transform,
                  rhat=rhat,
                  **kwargs)
    g = plt.gca()
    #g.set(xlim=(None, None))
    if vline_label is not None:
        no_effect = float(transform(0))
        g.axes.axvline(no_effect, color='red')
        g.axes.annotate(vline_label, [no_effect, -.5], rotation=90, va='center', ha='right', color='red')
    return g


def posterior_mode(trace):
    """Get posterior mode of a PyMC3 trace.

    Uses SciPy's Gaussian Kernel Density Estimation to fit density functions to the trace posteriors.

    :param trace: PyMC3 trace object
    :returns: numpy array of trace variable modes
    """
    def _posterior_mode(samples, bins=500):
        samples = samples[np.isfinite(samples)]
        xmin = np.min(samples)
        xmax = np.max(samples)
        kernel = scipy.stats.gaussian_kde(samples)
        density = kernel.pdf(np.linspace(xmin, xmax, bins))
        step = (xmax - xmin) / bins
        return xmin + np.argmax(density) * step
    return np.apply_along_axis(_posterior_mode, 0, trace)


def summary(trace, **kwargs):
    """Improve PyMC3 summary function by adding posterior mode.

    :param trace: PyMC3 trace object
    :param kwargs: keyword args for PyMC3 trace summary function
    :returns: PyMC3 trace summary in a pandas DataFrame
    """
    return pm.summary(trace,
                      extend=True,
                      stat_funcs=[lambda x: pd.Series(posterior_mode(x), name='mode')],
                      **kwargs)


def evidence_ratio_unsafe(hypothesis, trace):
    """Compute evidence ratio for a hypothesis about a single coefficient.

    This version is less safe because it uses an eval() call, but it's flexible.

    :param hypothesis: hypothesis in the form of an inequality such as 'Intercept > 0'
    :param trace: PyMC3 trace object
    :returns: evidence ratio for the hypothesis
    """
    h = hypothesis.replace(' ', '')

    if '<' in h:
        a, b = h.split('<')
    elif '>' in h:
        b, a = h.split('>')
    else:
        print('unknown operator (please use < or >)')  # TODO: raise error instead

    def parse_eq(eq):
        eq_list = []
        i = 0
        for j, c in enumerate(eq):
            if c in ['+', '-']:
                eq_list.append(eq[i:j])
                eq_list.append(eq[j])
                i = j + 1
        eq_list.append(eq[i:])
        eq_list = [f"trace['{item}']" if (item in trace.varnames) else item for item in eq_list]
        return eval(''.join(eq_list))

    a = parse_eq(a)
    b = parse_eq(b)
    return np.sum(a < b) / np.sum(a > b)


def evidence_ratio(hypothesis, trace):
    """Compute evidence ratio for a hypothesis about a single coefficient.

    This version is safe because it parses without using eval(), but it's very limited in the kinds of inequalities it can compute.

    :param hypothesis: hypothesis of the form 'Intercept > 0', only trace varnames, numbers, and greater than/smaller than operators are allowed
    :param trace: PyMC3 trace object
    :returns: evidence ratio for the hypothesis
    """
    a, operator, b = hypothesis.split(' ')
    a = trace[a] if a in trace.varnames else float(a)
    b = trace[b] if b in trace.varnames else float(b)

    # compute inequalities
    sumab = np.sum(a > b)
    sumba = np.sum(b > a)

    # replace zeroes with ones to avoid infinite and zero evidence ratios
    sumab = 1 if sumab == 0 else sumab
    sumba = 1 if sumba == 0 else sumba

    # compute evidence ratios
    if operator == '<':
        er = sumba / sumab
    elif operator == '>':
        er = sumab / sumba
    else:
        print('unknown operator (please use < or >)')
    return er


def remove_random(trace):
    """Remove random effects from a PyMC3 trace object.

    Does not alter the original trace object, but makes a deep copy instead.
    Identifies random effects by looking for the pipe ('|') operator.

    :param trace: PyMC3 trace object
    :returns: PyMC3 trace object without random effects
    """
    trace = copy.deepcopy(trace)
    random_vars = [varname for varname in trace.varnames if '|' in varname]
    for random_var in random_vars:
        trace.remove_values(random_var)
    return trace


def trace_to_df(trace):
    """Casts trace to long form pandas DataFrame.

    :param trace: PyMC3 trace object
    :returns: pandas DataFrame containing trace
    """
    trace_dict = {varname: trace[varname] for varname in trace.varnames}
    return pd.melt(pd.DataFrame(trace_dict), value_vars=trace.varnames)


def scale_trace(trace, factor):
    """Scale trace by a constant factor.

    Scales trace by a constant factor, for instance to reverse the effect of standardizing the dependent variable.
    Does not alter the original trace object, but makes a deep copy instead.

    :param trace: PyMC3 trace object
    :param factor: constant scaling factor
    :returns: PyMC3 trace object containing scaled traces
    """
    trace = copy.deepcopy(trace)
    effects = dict()
    for varname in trace.varnames:
        effects[varname] = trace[varname] * factor

    # overwrite copied traces
    trace.add_values(effects, overwrite=True)

    return trace


def marginal_effects(trace, simple_effects=None, interactions=None):
    """Computes marginal effects traces from a PyMC3 trace object.

    Does not alter the original trace object, but makes a deep copy instead.

    :param trace: PyMC3 trace object
    :param simple_effects: list of simple effects (dummy coded binary predictors)
    :param interaction: predictor to compute interactions with simple effects for (continuous or contrast-coded binary predictor)
    :returns: PyMC3 trace object containing marginal effects traces
    """
    trace = copy.deepcopy(trace)
    trace = remove_random(trace)
    effects = dict()
    for varname in trace.varnames:
        effects[varname] = np.zeros(trace['Intercept'].shape)

    # set intercept/reference value
    effects['Intercept'] += trace['Intercept']

    for effectname in effects.keys():
        for varname in trace.varnames:

            # compute marginal effects from simple effects
            if set(effectname.split(':')) <= set(simple_effects):

                # add intercept for reference condition to all simple effects
                if varname == 'Intercept':
                    effects[effectname] += trace[varname]

                # add simple effects to get marginal effects (per condition intercepts)
                elif set(varname.split(':')) <= set(effectname.split(':')):
                        effects[effectname] += trace[varname]

            # add interactions to get marginal effects (per condition slopes)
            else:
                for interaction in interactions:
                    if interaction in set(varname.split(':')):
                        if set(varname.split(':')) <= set(effectname.split(':')):
                                effects[effectname] += trace[varname]

    # overwrite simple/main effect traces with marginal effect traces
    trace.add_values(effects, overwrite=True)

    return trace


def rename_vars(trace, varnames_dict):
    """Rename variables in a PyMC3 trace object.

    Does not alter the original trace object, but makes a deep copy instead.

    :param trace: PyMC3 trace object
    :param varnames_dict: dictionary with old varnames as keys and new varnames as corresponding values
    :returns: PyMC3 trace object with renamed variables
    """
    trace = copy.deepcopy(trace)
    for old_name, new_name in varnames_dict.items():
        trace.add_values({new_name: trace[old_name]})
        trace.remove_values(old_name)
    return trace


def plot_marginal(effects, rows=None, cols=None, hues=None, interaction=None, main=None, xlim=(-1, 1), alpha=.15, **kwargs):
    """Plot marginal effects.

    :param effects: marginal effects object (PyMC3 traces converted to marginal effects using the marginal_effects() function)
    :param rows: condition levels to assign to rows
    :param cols: condition levels to assign to columns
    :param hues: condition levels to assign to hues
    :param interaction: the interaction of interest
    :param xlim: the limits of the x axes, -1 to +1 by default
    :param alpha: opacity of shaded HPD area around the mode line
    """
    df = summary(effects)
    df['effect'] = df.index
    hue = None
    col = None
    row = None

    if hues is not None:
        df['hue'] = df['effect'].apply(lambda x: next(iter(set(hues) & set(x.split(':')))) if ':' in x else None)
        hue = 'hue'
    if rows is not None:
        df['row'] = df['effect'].apply(lambda x: next(iter(set(rows) & set(x.split(':')))) if ':' in x else None)
        row = 'row'
    if cols is not None:
        df['col'] = df['effect'].apply(lambda x: next(iter(set(cols) & set(x.split(':')))) if ':' in x else None)
        col = 'col'

    if None not in [hue, row, col]:
        df['marginal'] = df['hue'] + ':' + df['row'] + ':' + df['col']
    elif None not in [hue, row]:
        df['marginal'] = df['hue'] + ':' + df['row']
    elif None not in [hue, col]:
        df['marginal'] = df['hue'] + ':' + df['col']
    elif None not in [col, row]:
        df['marginal'] = df['row'] + ':' + df['col']

    if interaction is not None:
        df['is_slope'] = df['effect'].apply(lambda x: 1 if interaction in x else 0)
    elif main is not None:
        df['is_slope'] = df['effect'].apply(lambda x: 1 if x == main else 0)

    intercepts = df.loc[df['is_slope'] == 0]
    slopes = df.loc[df['is_slope'] == 1][['marginal', 'mode', 'hpd_2.5', 'hpd_97.5']]
    slopes = slopes.add_prefix('slope_')

    if main is not None:
        intercepts['marginal'] = np.nan
    df = intercepts.merge(slopes, left_on='marginal', right_on='slope_marginal')

    def plot_mode(intercept_mode, slope_mode, xlim=(-1, 1), **kwargs):
        x = np.linspace(*xlim, 200)
        y = slope_mode.values * x + intercept_mode.values
        return plt.plot(x, y, **kwargs)

    def plot_ci(intercept_lower, intercept_upper, slope_lower, slope_upper, xlim=(-1, 1), alpha=alpha, **kwargs):
        x = np.linspace(*xlim, 200)
        y_lower = np.min([slope_upper.values * x + intercept_lower.values, slope_lower.values * x + intercept_lower.values], axis=0)
        y_upper = np.max([slope_lower.values * x + intercept_upper.values, slope_upper.values * x + intercept_upper.values], axis=0)
        return plt.fill_between(x, y_lower, y_upper, alpha=alpha, **kwargs)

    g = sns.FacetGrid(hue=hue,
                      col=col,
                      row=row,
                      col_order=cols,
                      row_order=rows,
                      hue_order=hues,
                      data=df,
                      margin_titles=True,
                      xlim=xlim,
                      **kwargs)

    g.map(plot_ci, 'hpd_2.5', 'hpd_97.5', 'slope_hpd_2.5', 'slope_hpd_97.5', xlim=xlim)
    g.map(plot_mode, 'mode', 'slope_mode', xlim=xlim)
    if cols is not None:
        for i in range(g.axes.shape[1]):
            g.axes[0][i].set(title=cols[i])
    for i in range(g.axes.shape[0]):
        for j in range(g.axes.shape[1]):
            g.axes[i][j].axhline(0, color='black', linestyle=':')
    xlabel = interaction if interaction is not None else main
    g.set_axis_labels(xlabel, 'marginal effect')
    g.add_legend(title='')
    return g


def compare(models, labels=None, insample_dev=False, **kwargs):
    """Easier model comparison for BAMBI models

    Automatically expands model terms into formulas and sets them as model names

    :param models: list of BAMBI model objects
    :param kwargs: keyword args for PyMC3 model comparison function
    :returns: tuple of matplotlib figure object of model comparison and pandas DataFrame of model statistics
    """
    if labels is not None:
        for i in range(len(models)):
            models[i].backend.model.name = labels[i]
    else:
        for model in models:
            model.backend.model.name = ' + '.join(model.terms.keys())
    models = {model.backend.model: model.backend.trace for model in models}
    comparison = pm.compare(models, **kwargs)
    g = pm.compareplot(comparison, insample_dev=insample_dev)
    return g, comparison


def qqplot(predicted, observed, n=101):
    """Quantile-quantile plot for model fit evaluation

    Uses numpy quantiles and seaborn scatterplot to display quantile-quantile correspondence
    between predicted and observed values and overlays a 45 degree line to aid visual inspection

    :param predicted: numpy array of predicted values
    :param observed: numpy array of observed values
    :param n: number of quantiles to plot, defaults to 100 (percentiles)
    :returns: matplotlib figure object containing the qqplot
    """
    n = np.linspace(0, 100, n)
    x = np.percentile(predicted, n)
    y = np.percentile(observed, n)
    g = sns.scatterplot(x, y)
    lower = np.min([x, y])
    upper = np.max([x, y])
    g.axes.plot((lower, upper), (lower, upper), color='red')
    g.set(title='quantile-quantile plot', xlabel='predicted', ylabel='observed')
    return g
