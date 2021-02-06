from pynwb.behavior import Position
from ipywidgets import widgets, Layout, fixed
from nwbwidgets.utils.timeseries import align_by_times
from nwbwidgets.utils.widgets import interactive_output
from ndx_events import Events

import scipy
import numpy as np
import matplotlib.pyplot as plt


class JointPosPSTHWidget(widgets.VBox):
    def __init__(self, events: Events, position: Position,
                 sigma_in_secs=.05, ntt=1000):

        super().__init__()

        before_ft = widgets.FloatText(.5, min=0, description='before (s)', layout=Layout(width='200px'))
        after_ft = widgets.FloatText(2., min=0, description='after (s)', layout=Layout(width='200px'))

        starts = events.timestamps[:] - before_ft.value
        stops = events.timestamps[:] + after_ft.value

        # Extract reach arm label from events, format to match key in spatial series
        reach_arm = events.description
        reach_arm = map(lambda x: x.capitalize(), reach_arm.split('_'))
        reach_arm = list(reach_arm)
        reach_arm = '_'.join(reach_arm)
        spatialseries = position.spatial_series[reach_arm]
        self.unit = spatialseries.unit

        self.trials = align_by_times(spatialseries, starts, stops)
        print(self.trials)
        print(np.shape(self.trials))
        if self.trials is None:
            self.children = [widgets.HTML('No trials present')]
            return

        # self.gas = self.make_group_and_sort(window=False, control_order=False)

        self.controls = dict(
            trials=fixed(self.trials),
            ntt=fixed(ntt),
            after=after_ft,
            before=before_ft,
            # gas=self.gas,
            # progress_bar=fixed(progress_bar)
        )

        out_fig = interactive_output(self.trials_psth, self.controls)

        self.children = [
            widgets.HBox([
                widgets.VBox([
                    before_ft,
                    after_ft,
                ])
            ]),
            out_fig
        ]

    def trials_psth(self, trials=None, ntt=1000, before=0., after=1., figsize=(7, 7)):
        """

        Parameters
        ----------
        trials: array-like
            Array of trials aligned to events
        before: float
            Time before that event (should be positive)
        after: float
            Time after that event
        ntt:
            Number of time points to use for smooth curve

        figsize: tuple, optional

        Returns
        -------
        matplotlib.Figure

        """
        fig, axs = plt.subplots(figsize=figsize)
        print(axs)
        axs.set_title('PSTH for Joint Position')

        self.show_psth_smoothed(trials, axs, before, after, ntt=ntt)
        return fig

    def show_psth_smoothed(self, trials, ax, before, after, ntt=1000,
                           align_line_color=(.7, .7, .7)):
        if not len(trials):  # TODO: when does this occur?
            return
        print(all_data)
        all_data = np.hstack(trials)
        print(all_data)
        if not len(all_data):  # no spikes
            return
        tt = all_data
        group_stats = []
        this_mean = np.nanmean(all_data, axis=0)
        err = scipy.stats.sem(all_data, axis=0, nan_policy='omit')
        group_stats.append(
            dict(mean=this_mean,
                 lower=this_mean - 2 * err,
                 upper=this_mean + 2 * err,)
        )
        ax.plot(tt, group_stats['mean'])
        ax.fill_between(tt, group_stats['lower'], group_stats['upper'], alpha=.2)
        ax.set_xlim([-before, after])
        ax.set_ylabel('Joint Position {}'.format(self.unit))
        ax.set_xlabel('time (s)')
        ax.axvline(color=align_line_color)
