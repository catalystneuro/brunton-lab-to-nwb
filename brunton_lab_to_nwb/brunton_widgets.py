from pynwb.behavior import Position
from pynwb.ecephys import ElectricalSeries
from pynwb.file import NWBFile
from ipywidgets import widgets, Layout

from nwbwidgets.ecephys import ElectricalSeriesWidget
from nwbwidgets.utils.timeseries import align_by_times, get_timeseries_tt
from nwbwidgets.controllers import StartAndDurationController
from nwbwidgets.utils.widgets import interactive_output
from nwbwidgets.brains import HumanElectrodesPlotlyWidget
from ndx_events import Events

import plotly.graph_objects as go

import numpy as np
import matplotlib.pyplot as plt


class BruntonDashboard(widgets.VBox):
    def __init__(self, nwb_file: NWBFile):
        box_layout = Layout(justify_content='space-between',
                            align_content='space-between',
                            align_items='center',
                            flex_basis='auto',
                            )
        super().__init__(layout=box_layout)

        # Start time and duration controller
        self.tt = get_timeseries_tt(self.spatial_series, istart=self.spatial_series.starting_time)
        self.time_window_controller = StartAndDurationController(
            tmin=tt[0],
            tmax=tt[-1],
            start=0,
            duration=5,
        )
        frame_ind = np.searchsorted(self.tt, self.time_window_controller.start)

        self.position = nwb_file.processing['behavior'].data_interfaces['Position']
        self.events = nwb_file.processing['behavior'].data_interfaces['ReachEvents']

        self.psth_widget = JointPosPSTHWidget(self.events, self.position,
                                              foreign_time_window_controller = self.time_window_controller)
        self.brains_widget = HumanElectrodesPlotlyWidget(nwb_file.electrodes)
        self.ecog_widget = ElectricalSeriesWidget(nwb_file.acquisition['ElectricalSeries'],
                                                  foreign_time_window_controller = self.time_window_controller)
        self.skeleton_widget = SkeletonPlot(nwb_file.processing['behavior'].data_interfaces['Position'],
                                            frame_ind)

        # Updates list of valid spike times at each change in time range
        self.time_window_controller.observe(self.updated_time_range)

        self.children = [widgets.HBox([psth_widget,
                                      ecog_widget
                                       ],
                                      layout=box_layout
                                      ),
                         brains_widget,
                        ]

    def updated_time_range(self, change=None):
        """Operations to run whenever time range gets updated"""
        self.skeleton_widget.fig.data = None
        new_frame_ind = np.searchsorted(self.tt, self.time_window_controller.value[0])
        self.skeleton_widget.plot_skeleton(self.position,
                                           new_frame_ind)

class SkeletonPlot(widgets.Hbox):
    def __init__(self, position: Position,
                 new_frame_ind):
        super().__init__()

        self.fig = go.FigureWidget()
        self.plot_skeleton(position, frame_ind)

        self.children = [self.fig]

    def plot_skeleton(self, position, frame_ind):

        l_ear = position['L_Ear'].data[frame_ind]
        l_elbow = position['L_Elbow'].data[frame_ind]
        l_shoulder = position['L_Shoulder'].data[frame_ind]
        l_wrist = position['L_Wrist'].data[frame_ind]
        nose = position['Nose'].data[frame_ind]
        r_ear = position['R_Ear'].data[frame_ind]
        r_elbow = position['r_elbow'].data[frame_ind]
        r_shoulder = position['r_shoulder'].data[frame_ind]
        r_wrist = position['r_wrist'].data[frame_ind]

        skeleton_vector = np.vstack([l_wrist,
                                     l_elbow,
                                     l_shoulder,
                                     l_ear,
                                     nose,
                                     r_ear,
                                     r_shoulder,
                                     r_elbow,
                                     r_wrist
                                     ]
                                    )

        self.fig.add_trace(
            go.Scatter(x = skeleton_vector[:,0],
                       y = skeleton_vector[:,1],
                       mode='lines+markers',
                       marker_color='blue',
                       marker_size=4,
                       )
        )


class JointPosPSTHWidget(widgets.HBox):
    def __init__(self, events: Events,
                 position: Position,
                 acquisition: ElectricalSeries = None):
        super().__init__()

        before_ft = widgets.FloatText(1.5, min=0, description='before (s)', layout=Layout(width='200px'))
        after_ft = widgets.FloatText(1.5, min=0, description='after (s)', layout=Layout(width='200px'))
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

        # self.fig = go.FigureWidget()
        # self.ecog_psth(acquisition)

        self.children = [widgets.HBox([widgets.VBox([before_ft,
                                                     after_ft]),
                                       out_fig
                                       ]
                                      ),
                         # widgets.Vbox([self.fig
                         #               ]
                         #              )
                         ]

    def ecog_psth(self, acquisition, before=1.5, after=1.5):

        starts = self.events - before
        stops = self.events + after

        trials = align_by_times(acquisition, starts, stops)

        # Discard bad ECoG segments:
            # Compute log-transformed spectral power density for
            # each 10 second EcoG segment and discard segments with power below 0
            # dB or abnormally high power at 115–125 Hz (>3 SD)
            # compared to all segments

        # Compute log-transformed, time frequency spectral power
        # with Morlet wavelets

        # Baseline subtract each segment using a
        # baseline defined as 1.5–1 seconds before each movement initiation event

        # self.fig.add_trace()


    def trials_psth(self, before=1.5, after=1.5, figsize=(12, 12)):
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
        tt = get_timeseries_tt(self.spatial_series, istart=self.spatial_series.starting_time)
        zero_ind = before * (1 / (tt[1] - tt[0]))
        diff_x = trials[:, :, 0].T - trials[:, int(zero_ind), 0]
        diff_y = trials[:, :, 1].T - trials[:, int(zero_ind), 1]

        diffs = np.dstack([diff_x, diff_y])
        distance = np.linalg.norm(diffs, axis=2)

        if trials is None:
            self.children = [widgets.HTML('No trials present')]
            return

        fig, axs = plt.subplots(1, 1, figsize=figsize)
        axs.set_title('Event-triggered Wrist Displacement')

        self.show_psth(distance, axs, before, after)
        return fig

    def show_psth(self, trials, ax, before, after, align_line_color=(.7, .7, .7)):
        """
        Calculate descriptive stats (mean, std) on trialed data and plot

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
        ax.set_ylabel('Joint Position (pixels)')
        ax.set_xlabel('time (s)')
        ax.axvline(color=align_line_color)

def process_ecog(acquisition):
    pass
    # Remove DC drift by subtracting the median voltage of each electrode

    # Zero artifacts with absolute voltage > 50 interquartile range [IQR].

    # Band-pass filter the data (1–200 Hz)

    # Notch filter at 60 Hz

    # Resample data to 500 Hz

    # Re-referenced to common median for each grid, strip, or depth electrode group

    # Remove electrodes with standard deviation (> 5 IQR) or kurtosis (> 10 IQR)
    # compared to the median value across channels


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
