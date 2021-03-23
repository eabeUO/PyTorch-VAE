import glob
import os
import pandas as pd
import numpy as np
from functools import partial
import subprocess

rootdir = os.path.expanduser('~/Research/FMEphys/')
# rootdir = os.path.expanduser('~/niell/seuss/Research/FMEphys')
# Set up partial functions for directory managing
join = partial(os.path.join,rootdir)


# TrainSet = ['012821_EE8P6LT_control_Rig2_fm1_WORLD',
#             '020821_EE12P1RN_control_Rig2_fm1_WORLD',
#             '021021_EE12P1RN_control_Rig2_fm1_WORLD',
#             '021121_EE12P1RN_control_Rig2_fm1_WORLD',
#             '021721_EE11P11LT_control_Rig2_fm2_WORLD',
#             '022321_EE13P2RT_control_Rig2_fm1_WORLD',
#             '031021_EE11P13LTRN_control_Rig2_fm1_WORLD',]
# ValSet = ['021121_EE12P1RN_control_Rig2_fm2_WORLD']

########## Checks if path exists, if not then creates directory ##########
def check_path(basepath, path):
    if path in basepath:
        return basepath
    elif not os.path.exists(os.path.join(basepath, path)):
        os.makedirs(os.path.join(basepath, path))
        print('Added Directory:'+ os.path.join(basepath, path))
        return os.path.join(basepath, path)
    else:
        return os.path.join(basepath, path)

def extract_frames_from_csv(csv_path):
    AllExps = pd.read_csv(csv_path)

    GoodExps = AllExps[(AllExps['Experiment outcome']=='good')].copy().reset_index()
    GoodExps = pd.concat((GoodExps[(GoodExps['Computer']=='kraken')][['Experiment date','Animal name','Computer','Drive']],
                        GoodExps[(GoodExps['Computer']=='v2')][['Experiment date','Animal name','Computer','Drive']]))
    GoodExps['Experiment date'] = pd.to_datetime(GoodExps['Experiment date'],infer_datetime_format=True,format='%m%d%Y').dt.strftime('%2m%2d%2y')
    GoodExps['Computer']=GoodExps['Computer'].str.capitalize()

    for n in range(len(GoodExps)):
        Comp=GoodExps['Computer'][n]
        Drive=GoodExps['Drive'][n]
        Date=GoodExps['Experiment date'][n]
        Ani=GoodExps['Animal name'][n]
        WorldPath = os.path.join(os.path.expanduser('~/'),Comp,Drive,'freely_moving_ephys/ephys_recordings',Date,Ani,'fm1','*WORLDcalib.avi')

        FM1Cam = glob.glob(WorldPath)
        if len(FM1Cam) > 0:
            SavePath = os.path.join(check_path(rootdir,os.path.basename(FM1Cam[0])[:-9]),'frame_%06d.png')
            subprocess.call(['ffmpeg', '-i', FM1Cam[0], '-vf','fps=30', '-vf','scale=128:128', SavePath])
    



def create_train_val_csv(TrainSet,ValSet):
    ExpDir = []
    DNum = []
    for exp in TrainSet:
        DataPaths = sorted(glob.glob(join(exp,'*.png')))
        print('{}: '.format(exp),len(DataPaths))
        for n in range(len(DataPaths)):
            ExpDir.append(DataPaths[n].split('/')[-2])
            DNum.append(DataPaths[n].split('/')[-1])

    df = pd.DataFrame({'BasePath':ExpDir,'FileName':DNum})
    df.to_csv(join('WC_Train_Data.csv'))

    print('Total Training Size: ', len(df))


    ExpDir = []
    DNum = []
    for exp in ValSet:
        DataPaths = sorted(glob.glob(join(exp,'*.png')))
        print('{}: '.format(exp),len(DataPaths))
        for n in range(len(DataPaths)):
            ExpDir.append(DataPaths[n].split('/')[-2])
            DNum.append(DataPaths[n].split('/')[-1])

    df = pd.DataFrame({'BasePath':ExpDir,'FileName':DNum})
    df.to_csv(join('WC_Val_Data.csv'))

    print('Total Validation Size: ', len(df))


if __name__ == '__main__':
    
    csv_path = os.path.expanduser('~/Research/FMEphys/Completed_experiment_pool.csv')

    extract_frames_from_csv(csv_path)
    TrainSet = sorted([os.path.basename(x) for x in glob.glob(join('*WORLD'))])
    valnum = np.random.randint(len(TrainSet))
    ValSet = [TrainSet[valnum]]
    TrainSet.pop(valnum)
    
    create_train_val_csv(TrainSet,ValSet)