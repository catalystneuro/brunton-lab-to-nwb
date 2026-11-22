import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from h5py import File
from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.data_utils import DataChunkIterator
from lazy_ops import DatasetView
from pynwb import NWBFile, NWBHDF5IO, TimeSeries
from pynwb.behavior import Position, SpatialSeries
from pynwb.ecephys import ElectricalSeries
from pynwb.file import Subject, TimeIntervals
from ndx_events import Events


SPECIAL_CHANNELS = (b"EOGL", b"EOGR", b"ECGL", b"ECGR")
SECONDS_PER_DAY = 86400

# The source times are the hospital's local wall-clock times. The source encodes them as if they were UTC, and the
# offset of the hospital's clock from UTC is unknown. So the times are decoded in UTC, which keeps the wall-clock date
# and time of day, and session_start_time is given with a +00:00 offset. The recording dates were removed to
# de-identify the participants (Peterson et al. 2022), and the dates in the source are in January 2000. The session
# description says the same.
TIME_ZONE_NOTE = (
    "All times are the hospital's local wall-clock times. The recording dates were removed to de-identify the "
    "participants, and the dates are shifted to January 2000. The offset of the times from UTC is unknown, so "
    "session_start_time is given with a +00:00 offset. The session starts at midnight of the recording day, and "
    "every time in the file counts from that midnight."
)

# Age and sex of each participant, from Table 2 of Peterson et al. 2022, "AJILE12: Long-term naturalistic human
# intracranial neural recordings and pose", Scientific Data 9, 184 (https://doi.org/10.1038/s41597-022-01280-y).
# Participant P01 is subject "01", and so on (the paper: "sub-01_ses-3_behavior+ecephys.nwb contains data from
# participant P01 on recording day 3"). The paper's column is "Gender"; it is given here as the subject's sex. The age
# is in years, as an ISO 8601 duration.
SUBJECTS = {
    "01": dict(age="P44Y", sex="M"),
    "02": dict(age="P20Y", sex="M"),
    "03": dict(age="P33Y", sex="M"),
    "04": dict(age="P19Y", sex="F"),
    "05": dict(age="P31Y", sex="F"),
    "06": dict(age="P37Y", sex="M"),
    "07": dict(age="P26Y", sex="M"),
    "08": dict(age="P33Y", sex="F"),
    "09": dict(age="P20Y", sex="M"),
    "10": dict(age="P34Y", sex="M"),
    "11": dict(age="P34Y", sex="F"),
    "12": dict(age="P22Y", sex="M"),
}


def get_subject(subject_id):
    """Return the Subject for a participant of AJILE12, with the age and sex of Table 2 of the paper."""
    if subject_id not in SUBJECTS:
        raise KeyError(f"subject {subject_id!r} is not in Table 2 of the AJILE12 paper: give its age and sex in SUBJECTS")
    return Subject(subject_id=subject_id, species="Homo sapiens", **SUBJECTS[subject_id])


def get_session_clock(start_timestamp):
    """Return the session start time and the offset of the first ECoG sample.

    The session starts at midnight of the recording day. The pose, the coarse behavioral labels and the reach
    features count from that midnight. The ECoG, EOG and ECG start at `start_timestamp`, and the reach events count
    from the first ECoG sample, so these get the offset.

    Parameters
    ----------
    start_timestamp: float
        The source's `start_timestamp`: the wall-clock time of the first ECoG sample, encoded as if it were UTC.

    Returns
    -------
    session_start_time: datetime
        Midnight of the recording day, with a +00:00 offset.
    offset: float
        Seconds from that midnight to the first ECoG sample (rounded to microseconds).
    """
    start_timestamp = float(start_timestamp)
    midnight = start_timestamp - start_timestamp % SECONDS_PER_DAY
    session_start_time = datetime.fromtimestamp(midnight, tz=timezone.utc)
    offset = round(start_timestamp - midnight, 6)
    return session_start_time, offset


def run_conversion(
    fpath_in="/Volumes/easystore5T/data/Brunton/subj_01_day_4.h5",
    fpath_out="/Volumes/easystore5T/data/Brunton/subj_01_day_4.nwb",
    events_path="C:/Users/micha/Desktop/Brunton Lab Data/event_times.csv",
    r2_path="/Volumes/easystore5T/data/Brunton/full_model_r2.npy",
    coarse_events_path="/Volumes/easystore5T/data/Brunton/coarse_labels_2",
    reach_features_path="/Volumes/easystore5T/data/Brunton/behavioral_features.csv",
    elec_loc_labels_path="elec_loc_labels.csv",
    special_chans=SPECIAL_CHANNELS,
    session_description=None,
):
    print(f"Converting {fpath_in}...")
    fname = os.path.split(os.path.splitext(fpath_in)[0])[1]
    _, subject_id, _, session = fname.split("_")

    file = File(fpath_in, "r")

    # midnight of the recording day, and the seconds from it to the first ECoG sample
    session_start_time, ecog_offset = get_session_clock(file["start_timestamp"][()])

    nwbfile = NWBFile(
        session_description=TIME_ZONE_NOTE
        if session_description is None
        else f"{session_description} {TIME_ZONE_NOTE}",
        identifier=str(uuid.uuid4()),
        session_start_time=session_start_time,
        subject=get_subject(subject_id),
        session_id=session,
    )

    # extract electrode groups
    file_elec_col_names = file["chan_info"]["axis1"][:]
    elec_data = file["chan_info"]["block0_values"]

    re_exp = re.compile("([ a-zA-Z]+)([0-9]+)")

    channel_labels_dset = file["chan_info"]["axis0"]

    group_names, group_nums = [], []
    for i, bytes_ in enumerate(channel_labels_dset):
        if bytes_ not in special_chans:
            str_ = bytes_.decode()
            res = re_exp.match(str_).groups()
            group_names.append(res[0])
            group_nums.append(int(res[1]))

    is_elec = ~np.isin(channel_labels_dset, special_chans)

    dset = DatasetView(file["dataset"]).lazy_transpose()

    # add special channels
    for kwargs in (
        dict(
            name="EOGL", description="Electrooculography for tracking saccades - left",
        ),
        dict(
            name="EOGR", description="Electrooculography for tracking saccades - right",
        ),
        dict(
            name="ECGL", description="Electrooculography for tracking saccades - left",
        ),
        dict(
            name="ECGR", description="Electrooculography for tracking saccades - right",
        ),
    ):
        if kwargs["name"].encode() in channel_labels_dset:
            nwbfile.add_acquisition(
                TimeSeries(
                    rate=file["f_sample"][()],
                    starting_time=ecog_offset,  # recorded with the ECoG
                    conversion=np.nan,
                    unit="V",
                    data=dset[
                        :, list(channel_labels_dset).index(kwargs["name"].encode())
                    ],
                    **kwargs,
                )
            )

    # add electrode groups
    df = pd.read_csv(elec_loc_labels_path)
    df_subject = df[df["subject_ID"] == "subj" + subject_id]
    electrode_group_descriptions = {
        row["label"]: row["long_name"] for _, row in df_subject.iterrows()
    }

    groups_map = dict()
    for group_name, group_description in electrode_group_descriptions.items():
        device = nwbfile.create_device(name=group_name)
        groups_map[group_name] = nwbfile.create_electrode_group(
            name=group_name,
            description=group_description,
            device=device,
            location="unknown",
        )

    # add required cols to electrodes table
    for row, group_name in zip(elec_data[:].T, group_names):
        nwbfile.add_electrode(
            x=row[file_elec_col_names == b"X"][0],
            y=row[file_elec_col_names == b"Y"][0],
            z=row[file_elec_col_names == b"Z"][0],
            imp=np.nan,
            location="unknown",
            filtering="250 Hz lowpass",  # text since NWB 2.4.0; float32 before was a schema bug (nwb-schema #478)
            group=groups_map[group_name],
        )

    # load r2 values to input into custom cols in electrodes table
    r2 = np.load(r2_path)
    low_freq_r2 = np.ravel(r2[int(subject_id) - 1, : len(group_names), 0])

    high_freq_r2 = np.ravel(r2[int(subject_id) - 1, : len(group_names), 1])

    # add custom cols to electrodes table
    elecs_dset = file["chan_info"]["block0_values"]

    def get_data(label):
        return elecs_dset[file_elec_col_names == label, :].ravel()[is_elec]

    [
        nwbfile.add_electrode_column(**kwargs)
        for kwargs in (
            dict(
                name="standard_deviation",
                description="standard deviation of each electrode's data for the entire recording period",
                data=get_data(b"SD_channels"),
            ),
            dict(
                name="kurtosis",
                description="kurtosis of each electrode's data for the entire recording period",
                data=get_data(b"Kurt_channels"),
            ),
            dict(
                name="median_deviation",
                description="median absolute deviation estimator for standard deviation for each electrode",
                data=get_data(b"standardizeDenoms"),
            ),
            dict(
                name="good",
                description="good electrodes",
                data=get_data(b"goodChanInds").astype(bool),
            ),
            dict(
                name="low_freq_R2",
                description="R^2 for low frequency band on each electrode",
                data=low_freq_r2,
            ),
            dict(
                name="high_freq_R2",
                description="R^2 for high frequency band on each electrode",
                data=high_freq_r2,
            ),
        )
    ]

    # confirm that electrodes table looks right
    # nwbfile.electrodes.to_dataframe()

    # add ElectricalSeries
    elecs_data = dset.lazy_slice[:, is_elec]
    n_bytes = np.dtype(elecs_data).itemsize

    nwbfile.add_acquisition(
        ElectricalSeries(
            name="ElectricalSeries",
            data=H5DataIO(
                data=DataChunkIterator(
                    data=elecs_data,
                    maxshape=elecs_data.shape,
                    buffer_size=int(5000 * 1e6) // elecs_data.shape[1] * n_bytes,
                ),
                compression="gzip",
            ),
            rate=file["f_sample"][()],
            starting_time=ecog_offset,  # the ECoG starts at start_timestamp
            conversion=1e-6,  # data is in uV
            electrodes=nwbfile.create_electrode_table_region(
                region=list(range(len(nwbfile.electrodes))),
                description="all electrodes",
            ),
        )
    )

    # add pose data
    pose_dset = file["pose_data"]["block0_values"]

    nwbfile.create_processing_module(name="behavior", description="pose data").add(
        Position(
            spatial_series=[
                SpatialSeries(
                    name=file["pose_data"]["axis0"][x_ind][:-2].decode(),
                    data=H5DataIO(
                        data=pose_dset[:, [x_ind, y_ind]], compression="gzip"
                    ),
                    reference_frame="unknown",
                    conversion=np.nan,
                    rate=30.0,
                    starting_time=0.0,  # the pose covers the whole day, from midnight
                )
                for x_ind, y_ind in zip(
                    range(0, pose_dset.shape[1], 2), range(1, pose_dset.shape[1], 2)
                )
            ]
        )
    )

    # add events
    events = pd.read_csv(events_path)
    mask = (events["Subject"] == int(subject_id)) & (
        events["Recording day"] == int(session)
    )
    events = events[mask]
    # "Event time" counts from the first ECoG sample
    timestamps = events["Event time"].values + ecog_offset
    events = events.reset_index()

    events = Events(
        name="ReachEvents",
        description=events["Event type"][0],  # Specifies which arm was used
        timestamps=timestamps,
        resolution=2e-3,  # resolution of the timestamps, i.e., smallest possible difference between timestamps
    )

    # add the Events type to the processing group of the NWB file
    nwbfile.processing["behavior"].add(events)

    # add coarse behavioral labels (one per pose sample, so on the pose clock: from midnight)
    event_fp = f"sub{subject_id}_fullday_{session}"
    full_fp = coarse_events_path + "//" + event_fp + ".npy"
    coarse_events = np.load(full_fp, allow_pickle=True)

    label, data = np.unique(coarse_events, return_inverse=True)
    transition_idx = np.where(np.diff(data) != 0)
    start_t = (
        nwbfile.processing["behavior"]
        .data_interfaces["Position"]["L_Wrist"]
        .starting_time
    )
    rate = nwbfile.processing["behavior"].data_interfaces["Position"]["L_Wrist"].rate
    times = np.divide(transition_idx, rate) + start_t  # 30Hz sampling rate
    max_time = (np.shape(coarse_events)[0] / rate) + start_t
    times = np.hstack([start_t, np.ravel(times), max_time])
    transition_labels = np.hstack([label[data[transition_idx]], label[data[-1]]])

    nwbfile.add_epoch_column(name="labels", description="Coarse behavioral labels")

    for start_time, stop_time, label in zip(times[:-1], times[1:], transition_labels):
        nwbfile.add_epoch(start_time=start_time, stop_time=stop_time, labels=label)

    # add additional reaching features
    reach_features = pd.read_csv(reach_features_path)
    mask = (reach_features["Subject"] == int(subject_id)) & (
        reach_features["Recording day"] == int(session)
    )
    reach_features = reach_features[mask]

    reaches = TimeIntervals(name="reaches", description="Features of each reach")
    reaches.add_column(
        name="Reach_magnitude_px", description="Magnitude of reach in pixels"
    )
    reaches.add_column(name="Reach_angle_degrees", description="Reach angle in degrees")
    reaches.add_column(
        name="Onset_speed_px_per_sec", description="Onset speed in pixels / second)"
    )
    reaches.add_column(
        name="Speech_ratio",
        description="rough estimation of whether someone is likely to be speaking "
        "based on a power ratio of audio data; ranges from 0 (no "
        "speech) to 1 (high likelihood of speech)h",
    )
    reaches.add_column(
        name="Bimanual_ratio",
        description="ratio of ipsilateral wrist reach magnitude to the sum of "
        "ipsilateral and contralateral wrist magnitudes; ranges from "
        "0 (unimanual/contralateral move only) to 1 (only ipsilateral"
        " arm moving); 0.5 indicates bimanual movement",
    )
    reaches.add_column(
        name="Bimanual_overlap",
        description="The amount of ipsilateral and contralateral wrist temporal"
        "overlap as a fraction of the entire contralateral movement"
        " duration",
    )
    reaches.add_column(
        name="Bimanual_class",
        description="binary feature that classifies each movement event as "
        "unimanual (0) or bimanual (1) based on how close in time a "
        "ipsilateral wrist movement started relative to each "
        "contralateral wrist movement events",
    )
    for row in reach_features.iterrows():
        row_data = row[1]
        start_time = row_data["Time of day (sec)"]  # counts from midnight
        stop_time = start_time + row_data["Reach duration (sec)"]
        reaches.add_row(
            start_time=start_time,
            stop_time=stop_time,
            Reach_magnitude_px=row_data["Reach magnitude (px)"],
            Reach_angle_degrees=row_data["Reach angle (degrees)"],
            Onset_speed_px_per_sec=row_data["Onset speed (px/sec)"],
            Speech_ratio=row_data["Speech ratio"],
            Bimanual_ratio=row_data["Bimanual ratio"],
            Bimanual_overlap=row_data["Bimanual overlap (sec)"],
            Bimanual_class=bool(row_data["Bimanual class"]),
        )

    nwbfile.add_time_intervals(reaches)

    # write NWB file
    with NWBHDF5IO(fpath_out, "w") as io:
        io.write(nwbfile)


def convert_dir(
    in_dir,
    events_path,
    r2_path,
    coarse_events_path,
    reach_features_path,
    elec_loc_labels_path,
    n_jobs=1,
    overwrite: bool = False,
):
    all_data_files = [x.stem for x in Path(in_dir).iterdir() if ".h5" in x.suffix]
    nwb_files = [x.stem for x in Path(in_dir).iterdir() if ".nwb" in x.suffix]

    if overwrite:
        in_files = [os.path.join(in_dir, f"{x}.h5") for x in all_data_files]
    else:
        in_files = [
            os.path.join(in_dir, f"{x}.h5")
            for x in all_data_files
            if x not in nwb_files
        ]
    out_files = [os.path.join(in_dir, f"{Path(x).stem}.nwb") for x in in_files]

    Parallel(n_jobs=n_jobs)(
        delayed(run_conversion)(
            fpath_in,
            fpath_out,
            events_path,
            r2_path,
            coarse_events_path,
            reach_features_path,
            elec_loc_labels_path,
        )
        for fpath_in, fpath_out in zip(in_files, out_files)
    )
