import time

import bqplot.pyplot as bqplt
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from ipywidgets import widgets, Layout
from matplotlib.colors import to_hex
from ndx_events import Events
from nwbwidgets.base import lazy_tabs
from nwbwidgets.brains import HumanElectrodesPlotlyWidget
from nwbwidgets.controllers import StartAndDurationController
from nwbwidgets.ecephys import ElectricalSeriesWidget
from nwbwidgets.timeseries import SingleTraceWidget, SeparateTracesPlotlyWidget
from nwbwidgets.utils.timeseries import (
    align_by_times,
    get_timeseries_tt,
    timeseries_time_to_ind,
    get_timeseries_in_units,
    get_timeseries_maxt,
    get_timeseries_mint,
)
from nwbwidgets.utils.widgets import interactive_output
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.colors import unlabel_rgb
from plotly.subplots import make_subplots
from pynwb.behavior import Position
from pynwb.ecephys import ElectricalSeries
from pynwb.file import NWBFile

POSITION_KEYS = [
            "L_Wrist",
            "L_Elbow",
            "L_Shoulder",
            "L_Ear",
            "Nose",
            "R_Ear",
            "R_Shoulder",
            "R_Elbow",
            "R_Wrist",
        ]


class BruntonDashboard(widgets.VBox):
    def __init__(self, nwb_file: NWBFile, tab1="local"):
        self.box_layout = Layout(
            justify_content="space-between",
            # align_content='space-between',
            align_items="center",
            flex_basis="auto",
        )
        super().__init__(layout=self.box_layout)

        self.row_layout = Layout(
            justify_content="space-between",
            # align_content='space-between',
            align_items="flex-start",
            flex_basis="auto",
        )

        in_dict = {
            "Time Trace Plots": self.tab1 if tab1 == "local" else self.tab_stream,
            "Event-triggered Plots": self.tab2,
        }
        tabs = lazy_tabs(in_dict, nwb_file)
        accordion = widgets.Accordion(children=[tabs], selected_index=None)
        accordion.set_title(0, "Brunton Dashboard")
        self.children = [accordion]

    def tab1(self, nwb_file):
        position_keys = list(
            nwb_file.processing["behavior"]
            .data_interfaces["Position"]
            .spatial_series.keys()
        )
        spatial_series = nwb_file.processing["behavior"].data_interfaces["Position"][
            position_keys[0]
        ]
        tt = get_timeseries_tt(spatial_series, istart=spatial_series.starting_time)
        time_trace_window_controller = StartAndDurationController(
            tmax=tt[-1], tmin=tt[0], start=0, duration=5
        )
        reach_arm = (
            nwb_file.processing["behavior"].data_interfaces["ReachEvents"].description
        )
        reach_arm = map(lambda x: x.capitalize(), reach_arm.split("_"))
        reach_arm = list(reach_arm)
        reach_arm = "_".join(reach_arm)

        jointpos_widget = AllPositionTracesPlotlyWidget(
            nwb_file.processing["behavior"].data_interfaces["Position"][reach_arm],
            foreign_time_window_controller=time_trace_window_controller,
        )
        jointpos_label = widgets.Label("Movement segments")
        jointpos = widgets.VBox(
            [jointpos_label, jointpos_widget], layout=self.box_layout
        )

        skeleton_widget = SkeletonPlot(
            nwb_file.processing["behavior"].data_interfaces["Position"],
            foreign_time_window_controller=time_trace_window_controller,
        )
        skeleton_label = widgets.Label("Tracked joints")
        skeleton = widgets.VBox(
            [skeleton_label, skeleton_widget], layout=self.box_layout
        )

        ecog_widget = ElectricalSeriesWidget(
            nwb_file.acquisition["ElectricalSeries"],
            foreign_time_window_controller=time_trace_window_controller,
        )
        ecog_label = widgets.Label("Raw ECoG")
        ecog = widgets.VBox([ecog_label, ecog_widget], layout=self.box_layout)

        brain_widget = HumanElectrodesPlotlyWidget(nwb_file.electrodes)
        brain_label = widgets.Label("Subject electrode locations")
        brain = widgets.VBox([brain_label, brain_widget], layout=self.box_layout)

        tab1_hbox_header = widgets.HBox([time_trace_window_controller])

        tab1_row1_widgets = widgets.HBox(
            [
                skeleton,
                jointpos,
            ],
            layout=self.row_layout,
        )
        tab1_row2_widgets = widgets.HBox(
            [
                brain,
                ecog,
            ],
            layout=self.row_layout,
        )
        tab1 = widgets.VBox(
            [tab1_hbox_header, tab1_row1_widgets, tab1_row2_widgets],
            layout=self.box_layout,
        )
        return tab1

    def tab_stream(self, nwb_file):
        position_keys = list(
            nwb_file.processing["behavior"]
            .data_interfaces["Position"]
            .spatial_series.keys()
        )
        spatial_series = nwb_file.processing["behavior"].data_interfaces["Position"][
            position_keys[0]
        ]
        tt = get_timeseries_tt(spatial_series, istart=spatial_series.starting_time)
        time_trace_window_controller = StartAndDurationController(
            tmax=tt[-1], tmin=tt[0], start=0, duration=5
        )
        reach_arm = (
            nwb_file.processing["behavior"].data_interfaces["ReachEvents"].description
        )
        reach_arm = map(lambda x: x.capitalize(), reach_arm.split("_"))
        reach_arm = list(reach_arm)
        reach_arm = "_".join(reach_arm)

        jointpos_widget = SeparateTracesPlotlyWidget(
            nwb_file.processing["behavior"].data_interfaces["Position"][reach_arm],
            foreign_time_window_controller=time_trace_window_controller,
        )
        text = "(b) Movement segments"
        jointpos_label = widgets.HTML(value=f"<b><font size=6>{text}</b>")
        jointpos = widgets.VBox(
            [jointpos_label, jointpos_widget], layout=self.box_layout
        )

        skeleton_widget = SkeletonPlot(
            nwb_file.processing["behavior"].data_interfaces["Position"],
            foreign_time_window_controller=time_trace_window_controller,
        )
        text =  "(a) Tracked joints"
        skeleton_label = widgets.HTML(value=f"<b><font size=6>{text}</b>")
        skeleton = widgets.VBox(
            [skeleton_label, skeleton_widget], layout=self.box_layout
        )

        ecog_widget = ElectricalSeriesWidget(
            nwb_file.acquisition["ElectricalSeries"],
            foreign_time_window_controller=time_trace_window_controller,
        )
        text = "(d) Raw ECoG"
        ecog_label = widgets.HTML(value=f"<b><font size=6>{text}</b>")
        ecog = widgets.VBox([ecog_label, ecog_widget], layout=self.box_layout)

        brain_widget = HumanElectrodesPlotlyWidget(nwb_file.electrodes)
        text = "(c) Subject electrode locations"
        brain_label = widgets.HTML(value=f"<b><font size=6>{text}</b>")
        brain = widgets.VBox([brain_label, brain_widget], layout=self.box_layout)

        tab1_hbox_header = widgets.HBox([time_trace_window_controller])
        tab1_row1_widgets = widgets.HBox(
            [
                skeleton,
                jointpos,
            ],
            layout=self.row_layout,
        )
        tab1_row2_widgets = widgets.HBox(
            [
                brain,
                ecog,
            ],
            layout=self.row_layout,
        )
        tab1 = widgets.VBox([tab1_hbox_header, tab1_row1_widgets, tab1_row2_widgets])
        return tab1

    def tab2(self, nwb_file):
        # spatial_series = nwb_file.processing['behavior'].data_interfaces['Position']['L_Ear']
        # tt = get_timeseries_tt(spatial_series, istart=spatial_series.starting_time)
        # event_trace_window_controller = StartAndDurationController(
        #     tmax=tt[-1],
        #     tmin=tt[0],
        #     start=0,
        #     duration=5
        # )
        eta_widget = ETAWidget(
            nwb_file.processing["behavior"].data_interfaces["ReachEvents"],
            nwb_file.processing["behavior"].data_interfaces["Position"],
            nwb_file.acquisition["ElectricalSeries"],
            # foreign_time_window_controller=event_trace_window_controller
        )

        # tab2_hbox_header = widgets.HBox([event_trace_window_controller])
        tab2_row1_widgets = widgets.HBox(
            [
                eta_widget,
            ],
            layout=self.box_layout,
        )

        tab2 = widgets.VBox([tab2_row1_widgets])
        return tab2


class AllPositionTracesPlotlyWidget(SingleTraceWidget):
    def set_out_fig(self):

        timeseries = self.controls["timeseries"].value
        time_window = self.controls["time_window"].value

        istart = timeseries_time_to_ind(timeseries, time_window[0])
        istop = timeseries_time_to_ind(timeseries, time_window[1])

        data, units = get_timeseries_in_units(timeseries, istart, istop)

        tt = get_timeseries_tt(timeseries, istart, istop)

        positions = self.timeseries.get_ancestor("Position")
        position_colors = {
            key: color for key, color in zip(POSITION_KEYS, DEFAULT_PLOTLY_COLORS)
        }
        data_dim = data.shape[1]
        subplot_titles = np.repeat(POSITION_KEYS, data_dim)
        if (len(data.shape) > 1) | len(POSITION_KEYS) > 1:
            self.out_fig = go.FigureWidget(
                make_subplots(
                    rows=len(POSITION_KEYS), cols=2, subplot_titles=subplot_titles
                )
            )
            self.out_fig["layout"].update(width=800, height=700)
            for k, key in enumerate(POSITION_KEYS):
                data = positions[key].data[:]
                color = position_colors[key]
                for i, (yy, xyz) in enumerate(zip(data.T, ("x", "y", "z"))):
                    self.out_fig.add_trace(
                        go.Scattergl(x=tt, y=yy, marker_color=color, showlegend=False),
                        row=k + 1,
                        col=i + 1,
                    )
                    if units:
                        yaxes_label = f"{xyz} ({units})"
                    else:
                        yaxes_label = xyz
                    self.out_fig.update_yaxes(
                        title_text=yaxes_label, row=k + 1, col=i + 1
                    )
                    self.out_fig.update_xaxes(
                        showticklabels=False, row=k + 1, col=i + 1
                    )
                self.out_fig["layout"]["annotations"][(k * data_dim) + i][
                    "text"
                ] = f"{key}"
            self.out_fig.update_xaxes(showticklabels=True, row=k + 1, col=i)
            self.out_fig.update_xaxes(showticklabels=True, row=k + 1, col=i + 1)
            self.out_fig.update_xaxes(title_text="time (s)", row=k + 1, col=i)
            self.out_fig.update_xaxes(title_text="time (s)", row=k + 1, col=i + 1)

        else:
            self.out_fig = go.FigureWidget()
            self.out_fig.add_trace(go.Scatter(x=tt, y=data))
            self.out_fig.update_xaxes(title_text="time (s)")

        def on_change(change):
            time_window = self.controls["time_window"].value
            istart = timeseries_time_to_ind(timeseries, time_window[0])
            istop = timeseries_time_to_ind(timeseries, time_window[1])

            tt = get_timeseries_tt(timeseries, istart, istop)
            yy, units = get_timeseries_in_units(timeseries, istart, istop)

            with self.out_fig.batch_update():
                if len(yy.shape) == 1:
                    self.out_fig.data[0].x = tt
                    self.out_fig.data[0].y = yy
                else:
                    for k, key in enumerate(POSITION_KEYS):
                        data = positions[key]
                        yy, units = get_timeseries_in_units(data, istart, istop)

                        for i, dd in enumerate(yy.T):
                            self.out_fig.data[(k * data_dim) + i].x = tt
                            self.out_fig.data[(k * data_dim) + i].y = dd

        self.controls["time_window"].observe(on_change)


class SkeletonPlot(widgets.VBox):
    def __init__(
        self,
        position: Position,
        foreign_time_window_controller: StartAndDurationController = None,
    ):
        super().__init__()

        self.position = position
        joint_keys = list(position.spatial_series.keys())
        self.joint_colors = []
        for (joint, c) in zip(joint_keys, DEFAULT_PLOTLY_COLORS):
            self.joint_colors.append(c)

        self.spatial_series = position.spatial_series[joint_keys[0]]
        if foreign_time_window_controller is None:
            self.time_window_controller = StartAndDurationController(
                tmax=get_timeseries_maxt(self.spatial_series),
                tmin=get_timeseries_mint(self.spatial_series),
                start=0,
                duration=5,
            )
            show_time_controller = True
        else:
            show_time_controller = False
            self.time_window_controller = foreign_time_window_controller
        frame_ind = timeseries_time_to_ind(self.spatial_series, self.time_window_controller.value[0])

        play_btn = widgets.Button(description="Start", icon="play")
        joint_colors = [to_hex(np.array(unlabel_rgb(x))/255)
                             for x in DEFAULT_PLOTLY_COLORS]
        self.joint_keys = POSITION_KEYS
        self.joint_colors = [
            joint_colors[0],
            joint_colors[1],
            joint_colors[2],
            joint_colors[9],
            joint_colors[4],
            joint_colors[3],
            joint_colors[5],
            joint_colors[4],
            joint_colors[9],
            joint_colors[6],
            joint_colors[7],
            joint_colors[8]
            ]
        self.skeleton_labels = [
            "L_Wrist",
            "L_Elbow",
            "L_Shoulder",
            "Neck",
            "Nose",
            "L_Ear",
            "R_Ear",
            "Nose",
            "Neck",
            "R_Shoulder",
            "R_Elbow",
            "R_Wrist",
        ]

        self.fig = bqplt.figure()  # animation_duration=int(1/spatial_series.rate*1000)
        self.plot_skeleton(frame_ind)

        # Updates list of valid spike times at each change in time range
        self.time_window_controller.observe(self.updated_time_range)
        play_btn.on_click(self.animate_scatter_chart)

        if show_time_controller:
            self.children = [
                self.time_window_controller,
                self.fig,
                play_btn
            ]
        else:
            self.children = [self.fig, play_btn]

    def updated_time_range(self, change=None):
        """Operations to run whenever time range gets updated"""
        if "new" in change:
            self.frame_ind_start = timeseries_time_to_ind(
                self.spatial_series, self.time_window_controller.value[0],
            )

            self.frame_ind_end = timeseries_time_to_ind(
                self.spatial_series, self.time_window_controller.value[1]
            )

            all_pos = np.vstack([
                x.data[self.frame_ind_start:self.frame_ind_end]
                for x in self.position.spatial_series.values()
            ])

            if not np.all(np.isnan(all_pos)):
                self.fig.axes[0].scale.min = np.nanmin(all_pos[:, 0])
                self.fig.axes[0].scale.max = np.nanmax(all_pos[:, 0]) + 30

                self.fig.axes[1].scale.max = np.nanmin(all_pos[:, 1])
                self.fig.axes[1].scale.min = np.nanmax(all_pos[:, 1])

            skeleton_vector = []
            for joint in self.joint_keys:
                skeleton_vector.append(self.position[joint].data[self.frame_ind_start])
            skeleton_vector = np.vstack(skeleton_vector)
            skeleton_vector = self.calc_centroid(skeleton_vector)

            with self.scat.hold_sync():
                self.scat.x = skeleton_vector[:, 0]
                self.scat.y = skeleton_vector[:, 1]
            with self.plot.hold_sync():
                self.plot.x = skeleton_vector[:, 0]
                self.plot.y = skeleton_vector[:, 1]

    def animate_scatter_chart(self, play_btn):

        sample_period = 1/list(self.position.spatial_series.values())[0].rate

        last_time = time.time()
        for frame_ind in range(self.frame_ind_start, self.frame_ind_end):
            skeleton_vector = []
            for joint in self.joint_keys:
                skeleton_vector.append(self.position[joint].data[frame_ind])
            skeleton_vector = np.vstack(skeleton_vector)
            skeleton_vector = self.calc_centroid(skeleton_vector)
            with self.scat.hold_sync():
                self.scat.x = skeleton_vector[:, 0]
                self.scat.y = skeleton_vector[:, 1]
            with self.plot.hold_sync():
                self.plot.x = skeleton_vector[:, 0]
                self.plot.y = skeleton_vector[:, 1]
            if time.time() - last_time < sample_period:
                time.sleep(sample_period - time.time() + last_time)
            last_time = time.time()

    def calc_centroid(self, skeleton_vector):
        base_of_neck = (skeleton_vector[2,:] + skeleton_vector[6,:])/2
        new_skeleton_vector = np.vstack(
            [skeleton_vector[0:3,:],
             base_of_neck,
             skeleton_vector[4,:], # nose
             skeleton_vector[3,:], # left ear
             skeleton_vector[5,:], # right ear
             skeleton_vector[4,:], # nose
             base_of_neck,
             skeleton_vector[6:]
             ])

        return new_skeleton_vector

    def plot_skeleton(self, frame_ind):

        skeleton_vector = []
        for joint in self.joint_keys:
            skeleton_vector.append(self.position[joint].data[frame_ind])

        skeleton_vector = np.vstack(skeleton_vector)
        skeleton_vector = self.calc_centroid(skeleton_vector)
        self.fig.layout.height = "500px"
        self.fig.layout.width = "600px"



        self.plot = bqplt.plot(
            x=skeleton_vector[:, 0],
            y=skeleton_vector[:, 1],
            colors=self.joint_colors,
            default_size=200,
            marker="circle",
            names=self.skeleton_labels,
        )

        self.scat = bqplt.scatter(
            x=skeleton_vector[:, 0],
            y=skeleton_vector[:, 1],
            colors=self.joint_colors,
            default_size=200,
            marker="circle",
            names=self.skeleton_labels,
        )

        bqplt.grids(value="none")

        # self.fig.add_trace(
        #     go.Scatter(x=skeleton_vector[:,0],
        #                y=-skeleton_vector[:,1],
        #                mode='lines+markers+text',
        #                marker_color=self.joint_colors,
        #                marker_size=12,
        #                text= joint_keys,
        #                hoverinfo='text',
        #                textposition="bottom center"
        #                )
        # )

        # options = {'color': dict(label='Category', orientation='vertical', side='right')}

        # self.fig.update_layout(
        #     xaxis = dict(
        #             showgrid=False,  # thin lines in the background
        #             zeroline=False,  # thick line at x=0
        #             visible=False,  # numbers below
        #             ),
        #     yaxis = dict(
        #             showgrid=False,  # thin lines in the background
        #             zeroline=False,  # thick line at x=0
        #             visible=False,  # numbers below
        #             )
        # )


class ETAWidget(widgets.VBox):
    def __init__(
        self,
        events: Events,
        position: Position,
        acquisition: ElectricalSeries = None,
        foreign_time_window_controller: StartAndDurationController = None,
    ):
        super().__init__()

        before_ft = widgets.FloatText(
            1.5, min=0, description="before (s)", layout=Layout(width="200px")
        )
        after_ft = widgets.FloatText(
            1.5, min=0, description="after (s)", layout=Layout(width="200px")
        )
        # Extract reach arm label from events, format to match key in Position
        # spatial series
        reach_arm = events.description
        reach_arm = map(lambda x: x.capitalize(), reach_arm.split("_"))
        reach_arm = list(reach_arm)
        reach_arm = "_".join(reach_arm)
        self.spatial_series = position.spatial_series[reach_arm]

        # if foreign_time_window_controller is None:
        self.tt = get_timeseries_tt(
            self.spatial_series, istart=self.spatial_series.starting_time
        )
        #     self.time_window_controller = StartAndDurationController(
        #         tmax=self.tt[-1],
        #         tmin=self.tt[0],
        #         start=0,
        #         duration=5
        #     )
        #     show_time_controller = True
        # else:
        #     self.time_window_controller = foreign_time_window_controller
        #     show_time_controller = False

        # Store events in object
        self.events = events.timestamps[:]

        self.controls = dict(
            after=after_ft,
            before=before_ft,
            # time_window=self.time_window_controller
        )

        out_fig = interactive_output(self.trials_psth, self.controls)
        # self.time_window_controller.observe(self.updated_time_range)

        # self.fig = go.FigureWidget()
        # self.ecog_psth(acquisition)
        # if show_time_controller:
        #     header_row = widgets.HBox([before_ft,
        #                                after_ft,
        #                                self.time_window_controller
        #                               ]
        #                              )
        # else:
        header_row = widgets.HBox(
            [
                before_ft,
                after_ft,
            ]
        )

        self.children = [header_row, out_fig]
        # widgets.Vbox([self.fig
        #               ]
        #              )

    def trials_psth(self, before=1.5, after=1.5, figsize=(6, 6)):  # time_window
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
        # mask = (self.events > time_window[0]) \
        #        & (self.events < time_window[1])
        # active_events = self.events[mask]
        starts = self.events - before
        stops = self.events + after

        trials = align_by_times(self.spatial_series, starts, stops)

        if trials.size == 0:
            return print("No trials present")

        tt = get_timeseries_tt(
            self.spatial_series, istart=self.spatial_series.starting_time
        )
        zero_ind = before * (1 / (tt[1] - tt[0]))
        if len(np.shape(trials)) == 3:
            diff_x = trials[:, :, 0].T - trials[:, int(zero_ind), 0]
            diff_y = trials[:, :, 1].T - trials[:, int(zero_ind), 1]
            diffs = np.dstack([diff_x, diff_y])
            distance = np.linalg.norm(diffs, axis=2)
        elif len(np.shape(trials)) == 2:
            diff_x = trials[:, :].T - trials[:, int(zero_ind)]
        elif len(np.shape(trials)) == 1:
            diff_x = trials

        fig, axs = plt.subplots(1, 1, figsize=figsize)
        axs.set_title("Event-triggered Wrist Displacement")

        self.show_psth(distance, axs, before, after)
        return fig

    def show_psth(self, trials, ax, before, after, align_line_color=(0.7, 0.7, 0.7)):
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
        tt = np.linspace(
            -before, after, int((before + after) * self.spatial_series.rate)
        )
        this_mean = np.nanmean(trials, axis=1)
        err = np.nanstd(trials, axis=1)
        group_stats = dict(
            mean=this_mean,
            lower=this_mean - err,
            upper=this_mean + err,
        )
        ax.plot(tt, group_stats["mean"])
        ax.fill_between(tt, group_stats["lower"], group_stats["upper"], alpha=0.2)
        ax.set_xlim([-before, after])
        ax.set_ylabel("Joint Position (pixels)")
        ax.set_xlabel("time (s)")
        ax.axvline(color=align_line_color)

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
        speed = np.hstack(
            (0, np.sqrt(np.sum(np.diff(pos.T) ** 2, axis=0)) / np.diff(pos_tt))
        )
    else:
        speed = np.hstack((0, np.sqrt(np.diff(pos.T) ** 2) / np.diff(pos_tt)))
    return speed
