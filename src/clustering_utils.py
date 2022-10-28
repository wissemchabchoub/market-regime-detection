"""Clustering Utils"""

import warnings

import numpy as np
import pandas as pd

from sklearn.utils import check_random_state

from numpy.lib.stride_tricks import sliding_window_view
import seaborn as sns
import matplotlib.pyplot as plt
import plotly_express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.utils import check_random_state

from src.wk_means import W


# ------------------------------------------------ #
# ------------------- Helpers -------------------- #
# ------------------------------------------------ #
def Q(x, j):
    """Computes the j_th order stat

    Parameters
    ----------
    x : array or list like
        Sample of the distribution
    j : int
        order

    Returns
    -------
    float
        the j_th order stat
    """
    assert j > 0
    return np.partition(np.asarray(x), j-1)[j-1]


def partition(stream, h1, h2, y_truth=None, return_lift=False, include_last_partition=False):
    """Generates a lift and atoms from a stream

    Parameters
    ----------
    stream : pd.Series
        Stream of data
    h1 : int
        window size
    h2 : int
        step size
    y_truth : pd.Series, optional
        grouth truth for each observation, by default None
    return_lift : bool, optional
        if True, returns the underlying lift, by default False
    include_last_partition : bool, optional
        if True, include the last partition (not labeled and its index is unknown), by default False

    """
    # compute atoms and lift of the stream
    include_last_parition = True

    i_index = pd.RangeIndex(stop=len(stream))
    if include_last_partition == False:
        data = sliding_window_view(stream, window_shape=h1)[::-h2][::-1][:-1]
        last_obs_date = sliding_window_view(i_index, window_shape=h1)[
            ::-h2][::-1][:, -1][:-1]
        dates = stream.index[last_obs_date + 1]

    elif include_last_partition == True:
        data = sliding_window_view(stream, window_shape=h1)[::-h2][::-1]
        last_obs_date = sliding_window_view(i_index, window_shape=h1)[
            ::-h2][::-1][:, -1]
        dates = stream.index[last_obs_date[:-1] + 1]
        dates = dates.append(stream.index[-1:])
        print("Last index is not lagged")
        warnings.warn("Last index is not lagged")

    cols = [f'r_{i}' for i in range(1, h1+1)]
    lift = pd.DataFrame(data, index=dates, columns=cols)

    # compute the atoms defining the emrical measures of each partition
    atoms = np.concatenate(lift.apply(lambda series: np.array([Q(series, j) for j in range(
        1, len(series)+1)]), axis=1).values, axis=0).reshape(lift.shape)
    atoms = pd.DataFrame(atoms, index=lift.index, columns=[
        f'alpha_{i}' for i in range(1, h1+1)])

    if y_truth is not None:
        y_truth = y_truth[lift.index]
        if return_lift:
            return atoms, lift, y_truth

        return atoms, y_truth

    if return_lift:
        return atoms, lift

    return atoms


# ------------------------------------------------ #
#------------ Plots for the notebook ------------- #
# ------------------------------------------------ #


def plot_regime_time_series(labels, ts, ts_secondary=None, return_df=False, return_fig=False):
    """generates a plot with a different color for each regime

    Parameters
    ----------
    labels : pd.Series
        labels (regime)
    ts : pd.Series
        original times series
    ts_secondary : ps.Series, optional
        another time series, by default None
    return_df : bool, optional
        if True, return a df, by default False
    return_fig : bool, optional
        if True, return the figure, by default False

    """
    n_clusters = len(labels.unique())

    df_plot_ = pd.Series(index=ts.index)
    df_plot_.loc[labels.index] = labels.values
    df_plot_ = df_plot_.ffill()

    df_plot = pd.DataFrame(index=ts.index, columns=[
                           f'regime {i}' for i in range(n_clusters)])
    for i in range(n_clusters):
        df_plot.loc[df_plot_ == i, f'regime {i}'] = ts.loc[df_plot_ == i]

    fig = df_plot.plot()
    fig.update_layout(xaxis_title='date',
                      yaxis_title=ts.name,)

    if ts_secondary is None and return_df:
        return df_plot
    else:
        if return_fig:
            return fig
        else:
            fig.show()

    if ts_secondary is not None:
        df_1 = ts_secondary.copy()
        df_1.index = df_1.index.normalize()

        df_2 = labels.copy()
        df_2.index = df_2.index.normalize()

        df_plot_ = pd.concat([df_1, df_2], axis=1).dropna()

        df_plot_ts_secondary = pd.DataFrame(index=df_plot_.index, columns=[
                                            f'regime {i}' for i in range(n_clusters)])
        for i in range(n_clusters):
            df_plot_ts_secondary.loc[df_plot_.regime == i, f'regime {i}'] = df_plot_[
                ts_secondary.name].loc[df_plot_.regime == i]

        fig = df_plot_ts_secondary.plot()
        fig.update_layout(xaxis_title='date',
                          yaxis_title=ts_secondary.name,)

        if return_df:
            return df_plot, df_plot_ts_secondary
        fig.show()

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        for k in range(n_clusters):

            fig.add_trace(
                go.Scatter(
                    y=df_plot[f'regime {k}'].values, x=df_plot.index, name=f"{ts.name} regime {k}"),
                secondary_y=False,
            )

            fig.add_trace(
                go.Scatter(y=df_plot_ts_secondary[f'regime {k}'].values,
                           x=df_plot_ts_secondary.index, name=f"{ts_secondary.name} regime {k}"),
                secondary_y=True,
            )

        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text=ts.name, secondary_y=False)
        fig.update_yaxes(title_text=ts_secondary.name, secondary_y=True)

        fig.show()


def target_analysis(labels, ts):
    """Generates an histogram of ts for each regime

    Parameters
    ----------
    labels : pd.Series
        labels (regimes)
    ts : pd.Series
        Time Series
    """
    df_1 = ts.copy()
    df_1.index = df_1.index.normalize()

    df_2 = labels.copy()
    df_2.index = df_2.index.normalize()

    df = pd.concat([df_1, df_2], axis=1).dropna()
    sns.histplot(df, x=ts.name, hue='regime', kde='True')
    plt.show()


def create_labels(est, features):
    """creates the labels from a trained estimator and a set of features

    Parameters
    ----------
    est : sklearn compatible estimator
        The estimator
    features : pd.DataFrame of shape (n_samples, n_features)
        The input samples.

    Returns
    -------
    pd.Series
        labels
    """
    labels = pd.Series(est.predict(features))
    labels.index = features.index
    labels.name = 'regime'
    return labels


def mean_volatility_plot(returns, labels, centroids, centroids_mean, centroids_vol):
    """Generates a plot of the returns in the mean volatility space
    """
    n_clusters = len(centroids)
    mean = returns.apply(lambda series: series.mean(), axis=1).values
    volatility = returns.apply(lambda series: series.std(), axis=1).values
    labels = labels.values

    mean = np.append(mean, centroids_mean)
    volatility = np.append(volatility, centroids_vol)
    labels = np.append(labels, ['red']*n_clusters)

    s = [len(labels)-i-1 for i in range(n_clusters)]
    c = labels
    #s = [closest(centroids[i], atoms.values) for i in range(n_clusters)]
    symbol = np.zeros(len(c))
    symbol[s] = 2
    size = np.ones(len(c))*0.05
    size[s] = 0.5
    #c[s] = 'red'

    fig = px.scatter(x=mean, y=volatility, color=c, symbol=symbol, size=size)
    fig.update_layout(xaxis_title='mean',
                      yaxis_title='volatility',)
    fig.show()


def plot_spread_evolution_up_to_regime_date(spread_data, labels, N, forward, start_date=None):
    """Plots spread evolution for each regime
        returns the cumsum(returns[t,t+N]]) or cumsum(returns[t-N,t]])
        * t ∈ Regime
        * t+N ot t-N are necessary in the Regime

    Parameters
    ----------
    spread_data : pd.Series
        time series of the spread
    labels : pd.Series
        labels (Regimes)
    N : int
        return period
    forward : boolean
        forward or backward returns
    """
    n_clusters = len(set(labels))
    if forward:
        spread_data = spread_data.diff(N).shift(-N)
    else:
        spread_data = spread_data.diff(N)

    spread_data = spread_data.loc[labels.index]
    
    if start_date is not None:
        spread_data = spread_data.loc[start_date:]
        
    #return (pd.concat([(spread_data[labels == i]).rename(f'regime {i}') for i in range(n_clusters)],
    #                 axis=1))

    fig = (pd.concat([(spread_data[labels == i]).rename(f'regime {i}') for i in range(n_clusters)],
                     axis=1).fillna(0).cumsum()/N).plot()

    fig.update_layout(xaxis_title='date',
                      yaxis_title='',
                      title='USHY OAS evolution up to regime date',)
    fig.show()


def plot_spread_evolion_within_each_regime(df, N,):
    """Plots spread evolution for each regime
        returns the cumsum(returns[t-N,t]])
        * t ∈ Regime
        * t-N ∈ Regime

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with each column having the values of a spectif regime (the rest are NaNs)
    N : int
        return period
    """
    fig = (df.diff(N).fillna(0)/N).cumsum().plot()
    fig.update_layout(xaxis_title='date',
                      yaxis_title='',
                      title='USHY OAS evolution within each regime',)
    fig.show()


def compute_WCSS_wk_means(estimator, atoms, labels):
    """computer WCSS for a WK means

    Parameters
    ----------
    estimator : WK means
        WK means
    atoms : pd.DataFrame
        features / atoms
    labels : pd.Series
        labels (Regimes)

    Returns
    -------
    floar
        WCSS
    """
    centroids = estimator.cluster_centers_
    S = 0
    for i in range(len(centroids)):
        c = centroids[i]
        for _, x in atoms[labels == i].iterrows():
            S += W(x, centroids[i])**2
    return S


def merton_jump_paths(T, steps, prng, mu_2, sigma_2,  lam_2, gamma_2, delta_2,
                      mu_1, sigma_1,  lam_1, gamma_1, delta_1,
                      regime_duration, n_regime_changes):
    """MJD process
    """
    dt = T/steps

    regime_dates = prng.random_integers(0, steps, n_regime_changes)

    mjd = []
    poi_rv = 0
    geo = 0
    size = 1
    for i in range(steps):
        if len((i - regime_dates)[(i - regime_dates) >= 0]) > 0:
            if (i - regime_dates)[(i - regime_dates) >= 0].min() <= regime_duration:
                mu, sigma,  lam, gamma, delta = mu_2, sigma_2,  lam_2, gamma_2, delta_2
            else:
                mu, sigma,  lam, gamma, delta = mu_1, sigma_1,  lam_1, gamma_1, delta_1
        else:
            mu, sigma,  lam, gamma, delta = mu_1, sigma_1,  lam_1, gamma_1, delta_1

        poi_rv += np.multiply(prng.poisson(lam*dt, size=size),
                              prng.normal(gamma, delta, size=size))

        geo += ((mu - sigma**2/2 - lam*(gamma + delta**2*0.5))*dt +
                sigma*np.sqrt(dt) *
                prng.normal(size=size))

        mjd.append(np.exp(geo+poi_rv))

    return np.array(mjd), regime_dates


# ------------------------------------------------ #
# --------------------- MMD ---------------------- #
# ------------------------------------------------ #

def mmd_base(x, y):
    n = len(x)
    m = len(y)
    mmd_1 = np.sum([kernel(x[i], x[j]) for i in range(n)
                   for j in range(n)])*1/n**2
    mmd_2 = np.sum([kernel(y[i], y[j]) for i in range(m)
                   for j in range(m)])*1/m**2
    mmd_3 = -np.sum([kernel(x[i], y[i]) for i in range(n)
                    for j in range(m)])*2/(m*n)

    return mmd_1 + mmd_2 + mmd_3


def kernel(x, y, sigma=0.1):
    return (x - y)**2 * ((2*sigma)**(-2))


def compute_mmd(X, labels, n_samples, clusters, random_state=None):
    random_state = check_random_state(random_state)
    X = np.array(X)
    mmd_s = []
    for i in tqdm(range(n_samples)):
        for cluster in clusters:
            seeds = random_state.permutation(len(X[labels == cluster]))[:2]
            sample = X[labels == cluster][seeds]
            mmd_s.append(mmd_base(sample[0], sample[1])**2)
    return mmd_s
