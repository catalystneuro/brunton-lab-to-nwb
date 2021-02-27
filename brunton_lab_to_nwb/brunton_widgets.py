from pynwb.behavior import Position
from ipywidgets import widgets, Layout, fixed
from nwbwidgets.utils.timeseries import align_by_times
from nwbwidgets.utils.widgets import interactive_output
from ndx_events import Events

import scipy
import numpy as np
import matplotlib.pyplot as plt


class JointPosPSTHWidget(widgets.HBox):
    def __init__(self, events: Events, position: Position):
        super().__init__()

        before_ft = widgets.FloatText(
            .5, min=0, description='before (s)', layout=Layout(width='200px'))
        after_ft = widgets.FloatText(
            2., min=0, description='after (s)', layout=Layout(width='200px'))
        # Extract reach arm label from events, format to match key in Position
        # spatial series
        reach_arm = events.description
        reach_arm = map(lambda x: x.capitalize(), reach_arm.split('_'))
        reach_arm = list(reach_arm)
        reach_arm = '_'.join(reach_arm)
        self.spatial_series = position.spatial_series[reach_arm]
        # Store events in object
        self.events = events.timestamps[:]
        self.controls = dict(after=after_ft, before=before_ft)

        out_fig = interactive_output(self.trials_psth, self.controls)

        self.children = [widgets.VBox([before_ft, after_ft]), out_fig]

    def trials_psth(self, before=0., after=1., figsize=(12, 12)):
        """
        Trial data by event times and plot

        Parameters
        ----------
        before: float
            Time before that event (should be positive)
        after: float
            Time after that event
        figsize: tuple, optional

        Returns
        -------
        matplotlib.Figure

        """
        starts = self.events - before
        stops = self.events + after
        # Construct time vector
        # tt_start = self.spatial_series.starting_time
        # num_pts = np.shape(self.spatial_series.data[:])[0]
        # tt_stop = num_pts / self.spatial_series.rate
        # tt = np.linspace(tt_start, tt_stop, int((num_pts)))
        # # Compute speed
        # speed = compute_speed(self.spatial_series.data[:], tt)
        # speed = np.reshape(speed, (-1, 1))
        # # Append to position data
        # pos_Wspeed = np.append(self.spatial_series.data[:], speed, axis=1)
        # del self.spatial_series.data
        # self.spatial_series.data = pos_Wspeed

        trials = align_by_times(self.spatial_series, starts, stops)
        if trials is None:
            self.children = [widgets.HTML('No trials present')]
            return

        fig, axs = plt.subplots(2, 1, figsize=figsize)
        axs[0].set_title('PSTH for Joint X-Position')
        axs[1].set_title('PSTH for Joint Y-Position')
        # axs[2].set_title('PSTH for Speed (m/s)')

        self.show_psth(trials[:, :, 0], axs[0], before, after)
        self.show_psth(trials[:, :, 1], axs[1], before, after)
        # self.show_psth(trials[:, :, 2], axs[2], before, after, tt)
        return fig

    def show_psth(self, trials, ax, before, after, align_line_color=(.7, .7, .7)):
        """
        Calculate descriptive stats (mean, sem) on trialed data and plot

        Parameters
        ----------
        trials: np.ndarray(dtype=float)
            Array with trialed data of shape=(n_trials, n_time, # data columns))
        ax: matplotlib.axes._subplots.AxesSubplot
            Subplot axes to plot figure
        before: float
            Time before that event (should be positive)
        after: float
            Time after that event
        align_line_color: tuple, optional

        Returns
        -------
        matplotlib.Figure

        """
        tt = np.linspace(-before, after, int((before + after) * self.spatial_series.rate))
        trials = trials.T - trials[:,0]
        this_mean = np.nanmean(trials, axis=1)
        err = scipy.stats.sem(trials, axis=1, nan_policy='omit')
        group_stats = dict(mean=this_mean,
                           lower=this_mean - 2 * err,
                           upper=this_mean + 2 * err,
                           )
        ax.plot(tt, group_stats['mean'])
        ax.fill_between(tt, group_stats['lower'], group_stats['upper'], alpha=.2)
        ax.set_xlim([-before, after])
        ax.set_ylabel('Joint Position ({})'.format(self.spatial_series.unit))
        ax.set_xlabel('time (s)')
        ax.axvline(color=align_line_color)


def compute_speed(pos, pos_tt):
    """Compute boolean of whether the speed of the animal was above a threshold
    for each time point

    Parameters
    ----------
    pos: np.ndarray(dtype=float)
        in meters
    pos_tt: np.ndarray(dtype=float)
        in seconds
    smooth_param: float, optional

    Returns
    -------
    running: np.ndarray(dtype=bool)

    """
    if len(pos.shape) > 1:
        speed = np.hstack((0, np.sqrt(np.sum(np.diff(pos.T) ** 2, axis=0)) / np.diff(pos_tt)))
    else:
        speed = np.hstack((0, np.sqrt(np.diff(pos.T) ** 2) / np.diff(pos_tt)))
    return speed
