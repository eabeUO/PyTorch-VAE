import glob, os
import pandas as pd
from functools import partial

rootdir = os.path.expanduser('~/Research/FMEphys/')
# rootdir = os.path.expanduser('~/niell/seuss/Research/FMEphys')
# Set up partial functions for directory managing
join = partial(os.path.join,rootdir)

DataDir = os.path.join(os.path.expanduser('~/Research/'),Comp,Drive,'freely_moving_ephys/ephys_recordings',Date,Ani,'fm1')

TrainSet = ['012821_EE8P6LT_control_Rig2_fm1_WORLD',
            '020821_EE12P1RN_control_Rig2_fm1_WORLD',
            '021021_EE12P1RN_control_Rig2_fm1_WORLD',
            '021121_EE12P1RN_control_Rig2_fm1_WORLD',
            '021721_EE11P11LT_control_Rig2_fm2_WORLD',
            '022321_EE13P2RT_control_Rig2_fm1_WORLD',
            '031021_EE11P13LTRN_control_Rig2_fm1_WORLD',]
ExpDir = []
DNum = []
for exp in TrainSet:
    DataPaths = sorted(glob.glob(join(exp,'*.png')))
    print('{}: '.format(exp),len(DataPaths))
    for n in range(len(DataPaths)):
        ExpDir.append(DataPaths[n].split('/')[-2])
        DNum.append(DataPaths[n].split('/')[-1])

df = pd.DataFrame({'BasePath':ExpDir,'FileName':DNum})

df.to_csv(join('WC_Train2_Data.csv'))

print('Total Training Size: ', len(df))


ValSet = ['021121_EE12P1RN_control_Rig2_fm2_WORLD']
ExpDir = []
DNum = []
for exp in ValSet:
    DataPaths = sorted(glob.glob(join(exp,'*.png')))
    print('{}: '.format(exp),len(DataPaths))
    for n in range(len(DataPaths)):
        ExpDir.append(DataPaths[n].split('/')[-2])
        DNum.append(DataPaths[n].split('/')[-1])

df = pd.DataFrame({'BasePath':ExpDir,'FileName':DNum})
df.to_csv(join('WC_Val2_Data.csv'))

print('Total Validation Size: ', len(df))
