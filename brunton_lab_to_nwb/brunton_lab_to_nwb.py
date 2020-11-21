import re
import uuid
from datetime import datetime

import numpy as np
from h5py import File
from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.data_utils import DataChunkIterator
from lazy_ops import DatasetView
from pynwb import NWBFile, NWBHDF5IO, TimeSeries
from pynwb.behavior import Position, SpatialSeries
from pynwb.ecephys import ElectricalSeries


def run_conversion(
        fpath_in='/Volumes/easystore5T/data/Brunton/subj_01_day_4.h5',
        out_fpath='/Volumes/easystore5T/data/Brunton/subj_01_day_4.nwb',
        special_chans=(b'EOGL', b'EOGR', b'ECGL', b'ECGR'),
        session_description='no description'
):
    file = File(fpath_in, 'r')

    nwbfile = NWBFile(
        session_description=session_description,
        identifier=str(uuid.uuid4()),
        session_start_time=datetime.fromtimestamp(file['start_timestamp'][()])
    )

    # extract electrode groups

    file_elec_col_names = file['chan_info']['axis1'][:]
    elec_data = file['chan_info']['block0_values']

    re_exp = re.compile("([a-zA-Z]+)([0-9]+)")

    channel_labels_dset = file['chan_info']['axis0']

    group_names, group_nums = [], []
    for i, bytes_ in enumerate(channel_labels_dset):
        if bytes_ not in special_chans:
            str_ = bytes_.decode()
            res = re_exp.match(str_).groups()
            group_names.append(res[0])
            group_nums.append(int(res[1]))

    is_elec = ~np.isin(channel_labels_dset, special_chans)

    dset = DatasetView(file['dataset']).lazy_transpose()

    # add special channels
    for kwargs in (
            dict(
                name='EOGL',
                description='Electrooculography for tracking saccades - left',
            ),
            dict(
                name='EOGR',
                description='Electrooculography for tracking saccades - right',
            ),
            dict(
                name='ECGL',
                description='Electrooculography for tracking saccades - left',
            ),
            dict(
                name='ECGR',
                description='Electrooculography for tracking saccades - right',
            )
    ):
        if kwargs['name'].encode() in special_chans:
            nwbfile.add_acquisition(
                TimeSeries(
                    rate=file['f_sample'][()],
                    conversion=np.nan,
                    unit='V',
                    data=dset[:, channel_labels_dset == kwargs['name'].encode()],
                    **kwargs
                )
            )

    # add electrode groups
    electrode_group_descriptions = dict(
        GRID='ECoG grid',
        LAT='lateral anterior temporal strip',
        LID='lateral inferior depth',
        LMT='lateral medial temporal strip',
        LPT='lateral posterior temporal strip',
        LTO='lateral temporal occipital strip',
    )

    groups_map = dict()
    for group_name, group_description in electrode_group_descriptions.items():
        device = nwbfile.create_device(name=group_name)
        groups_map[group_name] = nwbfile.create_electrode_group(
            name=group_name,
            description=group_description,
            device=device,
            location='unknown'
        )

    # add required cols to electrodes table
    for row, group_name in zip(elec_data[:].T, group_names):
        nwbfile.add_electrode(
            x=row[file_elec_col_names == b'X'][0],
            y=row[file_elec_col_names == b'Y'][0],
            z=row[file_elec_col_names == b'Z'][0],
            imp=np.nan,
            location='unknown',
            filtering='250 Hz lowpass',
            group=groups_map[group_name]
        )

    # add custom cols to electrodes table
    elecs_dset = file['chan_info']['block0_values']

    [nwbfile.add_electrode_column(**kwargs) for kwargs in (
        dict(
            name='standard_deviation',
            description="standard deviation of each electrode's data for the entire recording period",
            data=elecs_dset[file_elec_col_names == 'SD_channels', is_elec]
        ),
        dict(
            name='kurtosis',
            description="kurtosis of each electrode's data for the entire recording period",
            data=elecs_dset[file_elec_col_names == 'Kurt_channels', is_elec]
        ),
        dict(
            name='median_deviation',
            description="median absolute deviation estimator for standard deviation for each electrode",
            data=elecs_dset[file_elec_col_names == 'standardizeDenoms', is_elec]
        ),
        dict(
            name='good',
            description='good electrodes',
            data=elecs_dset[file_elec_col_names == 'goodChanInds', is_elec].astype(bool)
        )
    )]

    # confirm that electrodes table looks right
    # nwbfile.electrodes.to_dataframe()

    # add ElectricalSeries
    elecs_data = dset.lazy_slice[:, is_elec]

    nwbfile.add_acquisition(
        ElectricalSeries(
            name='ElectricalSeries',
            data=H5DataIO(
                data=DataChunkIterator(
                    data=elecs_data,
                    maxshape=elecs_data.shape,
                    buffer_size=int(1e4)
                ),
                compression='gzip'
            ),
            rate=file['f_sample'][()],
            electrodes=nwbfile.create_electrode_table_region(
                region=list(range(len(nwbfile.electrodes))),
                description='all electrodes'
            )
        )
    )

    # add pose data
    pose_dset = file['pose_data']['block0_values']

    nwbfile.create_processing_module(
        name='behavior',
        description='pose data').add(
        Position(
            spatial_series=[
                SpatialSeries(
                    name=file['pose_data']['axis0'][x_ind][:-2].decode(),
                    data=H5DataIO(
                        data=pose_dset[:, [x_ind, y_ind]],
                        compression='gzip'
                    ),
                    reference_frame='unknown',
                    conversion=np.nan,
                    rate=30.
                ) for x_ind, y_ind in zip(
                    range(0, pose_dset.shape[1], 2),
                    range(1, pose_dset.shape[1], 2))
            ]
        )
    )

    # write NWB file
    with NWBHDF5IO(out_fpath, 'w') as io:
        io.write(nwbfile)
