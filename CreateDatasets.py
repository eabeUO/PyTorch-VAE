import glob
import os
import pandas as pd
import argparse
import numpy as np
from functools import partial
import subprocess

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'False', 'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'True', 'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

parser = argparse.ArgumentParser(description='Create Dataset for WC Data')
parser.add_argument('--rootdir',  '-r',
                    help =  'RootDir',
                    default='~/Research/FMEphys/')
parser.add_argument('--DatasetType', type=str, default='3d')
parser.add_argument('--extract_frames', type=str_to_bool, nargs='?', const=True, default=False)
parser.add_argument('--N_fm',  '-n', type=int, default=16)
args = parser.parse_args()
rootdir = os.path.expanduser(args.rootdir)

##### Set up partial functions for directory managing
join = partial(os.path.join,rootdir)

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

########## Loads CSV with Experiments and extrats WC frames ##########
def extract_frames_from_csv(csv_path):
    AllExps = pd.read_csv(csv_path)

    GoodExps = AllExps[(AllExps['Experiment outcome']=='good')].copy().reset_index()
    GoodExps = pd.concat((GoodExps[(GoodExps['Computer']=='kraken')][['Experiment date','Animal name','Computer','Drive']],
                        GoodExps[(GoodExps['Computer']=='v2')][['Experiment date','Animal name','Computer','Drive']]))
    GoodExps['Experiment date'] = pd.to_datetime(GoodExps['Experiment date'],infer_datetime_format=True,format='%m%d%Y').dt.strftime('%2m%2d%2y')
    GoodExps['Computer']=GoodExps['Computer'].str.capitalize()
    print('Number of Experiments:', len(GoodExps))

    for n in range(len(GoodExps)):
        Comp=GoodExps['Computer'][n]
        Drive=GoodExps['Drive'][n]
        Date=GoodExps['Experiment date'][n]
        Ani=GoodExps['Animal name'][n]
        WorldPath = os.path.join(os.path.expanduser('~/'),'Seuss',Comp,Drive,'freely_moving_ephys/ephys_recordings',Date,Ani,'fm1','*WORLDcalib.avi')

        FM1Cam = glob.glob(WorldPath)
        if len(FM1Cam) > 0:
            SavePath = os.path.join(check_path(rootdir,os.path.basename(FM1Cam[0])[:-9]),'frame_%06d.png')
            subprocess.call(['ffmpeg', '-i', FM1Cam[0], '-vf','fps=30, scale=128:128', SavePath])
        else:
            print(Date+Ani,'No WORLDcalib.avi')

########## Creates csv, collecting frame paths for train and val datasets ##########
def create_train_val_csv(TrainSet,ValSet):
    ExpDir = []
    DNum = []
    for exp in TrainSet:
        DataPaths = sorted(glob.glob(join(exp,'*.png')))
        print('{}: '.format(exp),len(DataPaths))
        for n in range(len(DataPaths)):
            ExpDir.append(DataPaths[n].split('/')[-2])
            DNum.append(DataPaths[n].split('/')[-1])

    df_train = pd.DataFrame({'BasePath':ExpDir,'FileName':DNum})
    df_train.to_csv(join('WC_Train_Data.csv'))

    print('Total Training Size: ', len(df_train))


    ExpDir = []
    DNum = []
    for exp in ValSet:
        DataPaths = sorted(glob.glob(join(exp,'*.png')))
        print('{}: '.format(exp),len(DataPaths))
        for n in range(len(DataPaths)):
            ExpDir.append(DataPaths[n].split('/')[-2])
            DNum.append(DataPaths[n].split('/')[-1])

    df_val = pd.DataFrame({'BasePath':ExpDir,'FileName':DNum})
    df_val.to_csv(join('WC_Val_Data.csv'))

    print('Total Validation Size: ', len(df_val))
    return df_train, df_val

########## Creates csv, collecting frame paths for train and val datasets in shotgun style ##########
def create_train_val_csv_shotgun(TrainSet,ValSet,N_fm=4):
    ExpDir = []
    DNum = []
    for exp in TrainSet:
        DataPaths = sorted(glob.glob(join(exp,'*.png')))
        print('{}: '.format(exp),len(DataPaths))
        for n in range(len(DataPaths)):
            if n < N_fm: 
                DNum_temp = [DataPaths[0].split('/')[-1] for t in range(N_fm-n)]
                DNum_temp = sorted(DNum_temp + [DataPaths[n-t].split('/')[-1] for t in range(N_fm - len(DNum_temp))])
                DNum.append(DNum_temp)
            else:
                DNum.append([DataPaths[n+t-N_fm+1].split('/')[-1] for t in range(N_fm)])
            ExpDir.append(DataPaths[n].split('/')[-2])
    df_train = pd.DataFrame({'BasePath':ExpDir,'FileName':DNum})
    df_train.to_csv(join('WCShotgun_Train_Data.csv'))

    print('Total Training Size: ', len(df_train))


    ExpDir = []
    DNum = []
    for exp in ValSet:
        DataPaths = sorted(glob.glob(join(exp,'*.png')))
        print('{}: '.format(exp),len(DataPaths))
        for n in range(len(DataPaths)):
            if n < N_fm: 
                DNum_temp = [DataPaths[0].split('/')[-1] for t in range(N_fm-n)]
                DNum_temp = sorted(DNum_temp + [DataPaths[n-t].split('/')[-1] for t in range(N_fm - len(DNum_temp))])
                DNum.append(DNum_temp)
            else:
                DNum.append([DataPaths[n+t-N_fm+1].split('/')[-1] for t in range(N_fm)])
            ExpDir.append(DataPaths[n].split('/')[-2])
    df_val = pd.DataFrame({'BasePath':ExpDir,'FileName':DNum})
    df_val.to_csv(join('WCShotgun_Val_Data.csv'))

    print('Total Validation Size: ', len(df_val))
    return df_train, df_val

########## Creates csv, collecting frame paths for train and val datasets in 3d style ##########
def create_train_val_csv_3d(TrainSet,ValSet,N_fm=4):
    ExpDir = []
    DNum = []
    for exp in TrainSet:
        DataPaths = sorted(glob.glob(join(exp,'*.png')))
        print('{}: '.format(exp),len(DataPaths))
        for n in range(len(DataPaths)):
            if n < N_fm: 
                DNum_temp = [DataPaths[0].split('/')[-1] for t in range(N_fm-n)]
                DNum_temp = sorted(DNum_temp + [DataPaths[n-t].split('/')[-1] for t in range(N_fm - len(DNum_temp))])
                DNum.append(DNum_temp)
            else:
                DNum.append([DataPaths[n+t-N_fm+1].split('/')[-1] for t in range(N_fm)])
            ExpDir.append(DataPaths[n].split('/')[-2])
    df_train = pd.DataFrame({'BasePath':ExpDir,'FileName':DNum})
    df_train.to_csv(join('WC3d_Train_Data.csv'))

    print('Total Training Size: ', len(df_train))


    ExpDir = []
    DNum = []
    for exp in ValSet:
        DataPaths = sorted(glob.glob(join(exp,'*.png')))
        print('{}: '.format(exp),len(DataPaths))
        for n in range(len(DataPaths)):
            if n < N_fm: 
                DNum_temp = [DataPaths[0].split('/')[-1] for t in range(N_fm-n)]
                DNum_temp = sorted(DNum_temp + [DataPaths[n-t].split('/')[-1] for t in range(N_fm - len(DNum_temp))])
                DNum.append(DNum_temp)
            else:
                DNum.append([DataPaths[n+t-N_fm+1].split('/')[-1] for t in range(N_fm)])
            ExpDir.append(DataPaths[n].split('/')[-2])
    df_val = pd.DataFrame({'BasePath':ExpDir,'FileName':DNum})
    df_val.to_csv(join('WC3d_Val_Data.csv'))

    print('Total Validation Size: ', len(df_val))
    return df_train, df_val


if __name__ == '__main__':
    
    csv_path = os.path.expanduser('~/Research/Github/PyTorch-VAE/Completed_experiment_pool.csv')
    if args.extract_frames:
        extract_frames_from_csv(csv_path)
    TrainSet = sorted([os.path.basename(x) for x in glob.glob(join('*WORLD'))])
    valnum = np.random.randint(len(TrainSet))
    ValSet = [TrainSet[valnum]]
    TrainSet.pop(valnum)
    
    if args.DatasetType=='shotgun':
        df_train,df_val = create_train_val_csv_shotgun(TrainSet,ValSet,N_fm=args.N_fm)
    elif args.DatasetType=='3d':
        df_train,df_val = create_train_val_csv_3d(TrainSet,ValSet,N_fm=args.N_fm)
    else:
        df_train,df_val = create_train_val_csv(TrainSet,ValSet)