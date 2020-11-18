# brunton-lab-to-nwb

## Open questions for lab meeting are in straight braces []

## ToDo tasks are in curly {}
[] {Figure out how load continuous ECoG data}
[X] {Create an NWB file to write to}
[] {Create electrical series from ECoG data and write to NWB file}
    [] How to input electrode locations based on location matrix provide 
    [] What location to put for each electrode
[] {Input position}
[] {Input other features}


All ECoG and associated meta data are stored in fif files that are loaded with the MNE package into an MNE object.  
This MNE object contains behavioral data in a dataframe accessed via the .metadata attribute.

## NWBFile

## Subject

## Recording:
ECoG data is epoched. 
  ### Electrodes:
    - Electrode locations are stored in (MNE_object).info['chs'][(channel number)]['loc']
    

## Behavioral Features:
- 'day' - day of recording
- 'tod' - Time of Day of recording
- 'reach_duration' - total duration a reach lasted
- 'reach_r' - Reach magnitude
- 'reach_a' - Reach angle
- 'onset_velocity' - Onset speed of reach
- 'audio_ratio' - Speech ratio, whether subject was speaking during movement

### How much each wrist moved during movement:
  - 'I_over_C_ratio' - 
  - 'other_reach_overlap'- 
  - 'bimanual' - both wrists moved


## Epochs
