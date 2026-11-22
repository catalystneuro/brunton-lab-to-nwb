"""Check that every object of a converted file is on one clock: midnight of the recording day.

The source files are made up here, so these tests need no lab data.
"""
from datetime import datetime, timezone

import h5py
import numpy as np
import pandas as pd
import pytest
from pynwb import NWBHDF5IO

from brunton_lab_to_nwb import run_conversion
from brunton_lab_to_nwb.brunton_lab_to_nwb import get_session_clock

MIDNIGHT = datetime(2000, 1, 3, tzinfo=timezone.utc)
ECOG_RATE = 500.0
POSE_RATE = 30.0
N_ECOG_SAMPLES = 1000
N_POSE_SAMPLES = 60
ELECTRODES = [b"GRID1", b"GRID2", b"GRID3"]
SPECIAL = [b"EOGL", b"EOGR", b"ECGL", b"ECGR"]
POSE_PARTS = ["L_Wrist", "R_Wrist"]
EVENT_TIMES = [0.5, 1.25]  # seconds from the first ECoG sample
REACH_DURATION = 0.2


def make_source(folder, offset):
    """Write a made-up source file and its tables. The ECoG starts `offset` seconds after midnight."""
    fpath_in = folder / "subj_01_day_3.h5"
    channels = ELECTRODES + SPECIAL
    columns = [b"X", b"Y", b"Z", b"SD_channels", b"Kurt_channels", b"standardizeDenoms", b"goodChanInds"]
    rng = np.random.default_rng(0)
    with h5py.File(fpath_in, "w") as f:
        f["start_timestamp"] = MIDNIGHT.timestamp() + offset
        f["f_sample"] = ECOG_RATE
        f["dataset"] = rng.normal(size=(len(channels), N_ECOG_SAMPLES))
        f["chan_info/axis0"] = np.array(channels)
        f["chan_info/axis1"] = np.array(columns)
        f["chan_info/block0_values"] = np.ones((len(columns), len(channels)))
        f["pose_data/axis0"] = np.array([f"{part}_{xy}".encode() for part in POSE_PARTS for xy in "xy"])
        f["pose_data/block0_values"] = rng.normal(size=(N_POSE_SAMPLES, 2 * len(POSE_PARTS)))

    events_path = folder / "event_times.csv"
    pd.DataFrame(
        {"Subject": 1, "Recording day": 3, "Event time": EVENT_TIMES, "Event type": "l_wrist"}
    ).to_csv(events_path, index=False)

    # the reach features count from midnight
    reach_features_path = folder / "behavioral_features.csv"
    pd.DataFrame(
        {
            "Subject": 1,
            "Recording day": 3,
            "Time of day (sec)": [offset + t for t in EVENT_TIMES],
            "Reach duration (sec)": REACH_DURATION,
            "Reach magnitude (px)": 10.0,
            "Reach angle (degrees)": 45.0,
            "Onset speed (px/sec)": 5.0,
            "Speech ratio": 0.1,
            "Bimanual ratio": 0.5,
            "Bimanual overlap (sec)": 0.0,
            "Bimanual class": 0,
        }
    ).to_csv(reach_features_path, index=False)

    r2_path = folder / "full_model_r2.npy"
    np.save(r2_path, np.zeros((1, len(ELECTRODES), 2)))

    coarse_events_path = folder / "coarse_labels"
    coarse_events_path.mkdir()
    labels = np.array(["sleep"] * (N_POSE_SAMPLES // 2) + ["active"] * (N_POSE_SAMPLES // 2), dtype=object)
    np.save(coarse_events_path / "sub01_fullday_3.npy", labels, allow_pickle=True)

    elec_loc_labels_path = folder / "elec_loc_labels.csv"
    pd.DataFrame({"subject_ID": ["subj01"], "label": ["GRID"], "long_name": ["grid"]}).to_csv(
        elec_loc_labels_path, index=False
    )
    return dict(
        fpath_in=str(fpath_in),
        events_path=str(events_path),
        r2_path=str(r2_path),
        coarse_events_path=str(coarse_events_path),
        reach_features_path=str(reach_features_path),
        elec_loc_labels_path=str(elec_loc_labels_path),
    )


@pytest.mark.parametrize("offset", [0.0, 30048.322])
def test_session_clock(offset):
    session_start_time, ecog_offset = get_session_clock(MIDNIGHT.timestamp() + offset)
    assert session_start_time == MIDNIGHT
    assert session_start_time.utcoffset().total_seconds() == 0
    assert ecog_offset == offset


@pytest.mark.parametrize("offset", [0.0, 30048.322])
def test_one_clock(tmp_path, offset):
    paths = make_source(tmp_path, offset)
    fpath_out = tmp_path / "out.nwb"
    run_conversion(fpath_out=str(fpath_out), **paths)

    with NWBHDF5IO(str(fpath_out), "r", load_namespaces=True) as io:
        nwbfile = io.read()

        # the session starts at midnight of the recording day, in UTC
        assert nwbfile.session_start_time == MIDNIGHT
        assert nwbfile.session_start_time.utcoffset().total_seconds() == 0
        assert nwbfile.timestamps_reference_time == MIDNIGHT
        assert "wall-clock" in nwbfile.session_description

        # the ECoG, EOG and ECG start at the offset
        for name in ["ElectricalSeries", "EOGL", "EOGR", "ECGL", "ECGR"]:
            series = nwbfile.acquisition[name]
            assert series.starting_time == pytest.approx(offset, abs=1e-6)
            assert series.rate == ECOG_RATE

        # the reach events get the same offset
        behavior = nwbfile.processing["behavior"]
        reach_events = behavior["ReachEvents"].timestamps[:]
        np.testing.assert_allclose(reach_events, np.array(EVENT_TIMES) + offset, atol=1e-6)

        # the pose and the coarse labels are unchanged: they start at midnight
        for part in POSE_PARTS:
            assert behavior["Position"][part].starting_time == 0.0
            assert behavior["Position"][part].rate == POSE_RATE
        epochs = nwbfile.epochs.to_dataframe()
        assert epochs["start_time"].iloc[0] == 0.0
        assert epochs["stop_time"].iloc[-1] == pytest.approx(N_POSE_SAMPLES / POSE_RATE)
        assert list(epochs["labels"]) == ["sleep", "active"]
        assert epochs["start_time"].iloc[1] == pytest.approx((N_POSE_SAMPLES // 2 - 1) / POSE_RATE)

        # the reaches are unchanged, and now agree with the reach events
        reaches = nwbfile.intervals["reaches"].to_dataframe()
        np.testing.assert_allclose(reaches["start_time"], np.array(EVENT_TIMES) + offset, atol=1e-6)
        np.testing.assert_allclose(reaches["start_time"] - reach_events, 0.0, atol=1e-6)
        np.testing.assert_allclose(reaches["stop_time"] - reaches["start_time"], REACH_DURATION, atol=1e-6)
