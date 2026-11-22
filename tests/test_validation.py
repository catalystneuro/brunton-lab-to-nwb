"""Check that a converted file passes what `dandi validate` checks: the NWB schema (pynwb, against the namespaces
cached in the file) and the nwbinspector checks that the "dandi" configuration makes CRITICAL.

The source files are made up here (see test_time_bases.py), so these tests need no lab data.
"""
import pytest
from pynwb import NWBHDF5IO, validate

from brunton_lab_to_nwb import run_conversion
from brunton_lab_to_nwb.brunton_lab_to_nwb import SUBJECTS, get_subject

from test_time_bases import make_source

nwbinspector = pytest.importorskip("nwbinspector")


def test_subjects_from_the_paper():
    # Table 2 of the AJILE12 paper: 12 participants, 8 males and 4 females, 29.4 +/- 7.6 years old
    assert sorted(SUBJECTS) == [f"{i:02d}" for i in range(1, 13)]
    assert sum(s["sex"] == "M" for s in SUBJECTS.values()) == 8
    assert sum(s["sex"] == "F" for s in SUBJECTS.values()) == 4
    ages = [int(s["age"][1:-1]) for s in SUBJECTS.values()]
    assert round(sum(ages) / len(ages), 1) == 29.4
    subject = get_subject("04")
    assert (subject.age, subject.sex, subject.species) == ("P19Y", "F", "Homo sapiens")
    with pytest.raises(KeyError):
        get_subject("13")


@pytest.mark.parametrize("offset", [0.0, 30048.322])
def test_dandi_validation(tmp_path, offset):
    paths = make_source(tmp_path, offset)
    fpath_out = tmp_path / "sub-01_ses-3_behavior+ecephys.nwb"
    run_conversion(fpath_out=str(fpath_out), **paths)

    # the NWB schema, against the namespaces cached in the file (as dandi validate does)
    assert validate(path=str(fpath_out)) == []

    with NWBHDF5IO(str(fpath_out), "r", load_namespaces=True) as io:
        nwbfile = io.read()
        assert nwbfile.subject.age == "P44Y"
        assert nwbfile.subject.sex == "M"
        assert set(nwbfile.electrodes["filtering"][:]) == {"250 Hz lowpass"}

    messages = list(
        nwbinspector.inspect_nwbfile(
            nwbfile_path=str(fpath_out),
            skip_validate=True,
            config=nwbinspector.load_config(filepath_or_keyword="dandi"),
            importance_threshold=nwbinspector.Importance.CRITICAL,
        )
    )
    assert [m for m in messages if m is not None] == []
