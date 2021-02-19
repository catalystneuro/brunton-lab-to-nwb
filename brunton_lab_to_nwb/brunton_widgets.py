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

        before_ft = widgets.FloatText(.5, min=0, description='before (s)', layout=Layout(width='200px'))
        after_ft = widgets.FloatText(2., min=0, description='after (s)', layout=Layout(width='200px'))
        # Extract reach arm label from events, format to match key in Position
        # spatial series
        reach_arm = events.description
        reach_arm = map(lambda x: x.capitalize(), reach_arm.split('_'))
        reach_arm = list(reach_arm)
        reach_arm = '_'.join(reach_arm)
        self.spatial_series = position.spatial_series[reach_arm]
        # Store events in object
        self.events = events.timestamps[:]
        self.controls = dict(
            after=after_ft,
            before=before_ft,
        )

        out_fig = interactive_output(self.trials_psth, self.controls)

        self.children = [widgets.VBox([before_ft,
                                       after_ft,
                                       ]),
                         out_fig
                         ]

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

        trials = align_by_times(self.spatial_series, starts, stops)
        zero_ind = before * self.spatial_series.rate
        diff_x = trials[:, :, 0].T - trials[:, int(zero_ind), 0]
        diff_y = trials[:, :, 1].T - trials[:, int(zero_ind), 1]

        diffs = np.dstack([diff_x, diff_y])
        distance = np.linalg.norm(diffs, axis=2)

        if trials is None:
            self.children = [widgets.HTML('No trials present')]
            return

        fig, axs = plt.subplots(1, 1, figsize=figsize)
        axs.set_title('PSTH for Displacement')

        self.show_psth(distance, axs, before, after)
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
        this_mean = np.nanmean(trials, axis=1)
        err = np.nanstd(trials, axis=1)
        group_stats = dict(mean=this_mean,
                           lower=this_mean - err,
                           upper=this_mean + err,
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
