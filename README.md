# brunton-lab-to-nwb
Package for converting data in the Brunton Lab to NWB.

## Installation
```bash
pip install git+https://github.com/catalystneuro/brunton-lab-to-nwb.git
```

## Usage
```python
from brunton_lab_to_nwb import run_conversion

run_conversion('in/path', 'out/path')
```

## Example uses: 
### Load nwb file:
```python
from pynwb import NWBHDF5IO
io = NWBHDF5IO(r'C:\Users\micha\Desktop\Brunton Lab Data\H5\subj_01_day_3.nwb', mode='r')
nwb = io.read()
```

### See accessible fields in nwb file object:
```python
nwb_file.fields
```

### Get electrode series:
```python
nwb_file.electrodes
```

### Align events
```python
# Get reach events
events = nwb_file.processing["behavior"].data_interfaces["ReachEvents"]
# Get reach position
reach_arm_pos = nwb_file.processing["behavior"].data_interfaces["Position"]["L_Wrist"]
# Set window around event alignment
before = 1.5 # in seconds
after = 1.5 # in seconds
starts = self.events - before
stops = self.events + after
# Get trials aligned by events
trials = align_by_times(reach_arm_pos, starts, stops)
```
