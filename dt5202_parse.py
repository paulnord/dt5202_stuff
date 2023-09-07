import numpy as np
import pandas as pd
from datetime import datetime
import argparse

import uproot


def dt5202_text(file_name):
        with open(file_name, 'r') as file:
                tstamps = []
                trgids = []

                boards = []
                channels = []
                LGs = []
                HGs = []

                boards_evt = []
                channels_evt = []
                LGs_evt = []
                HGs_evt = []


                for line in file:
                        
                        if line.startswith('//'):
                                pass
                        elif 'Tstamp_us' in line:
                                pass
                        else:
                                tokens = line.split()
                                if len(tokens) == 6:
                                        if len(tstamps) > 0:
                                                boards.append(boards_evt)
                                                channels.append(channels_evt)
                                                LGs.append(LGs_evt)
                                                HGs.append(HGs_evt)
                                                
                                                boards_evt = []
                                                channels_evt = []
                                                LGs_evt = []
                                                HGs_evt = []

                                        tstamps.append(float(tokens[0]))
                                        trgids.append(int(tokens[1]))
                                        boards_evt.append(int(tokens[2]))
                                        channels_evt.append(int(tokens[3]))
                                        LGs_evt.append(int(tokens[4]))
                                        HGs_evt.append(int(tokens[5]))

                                elif len(tokens) == 4:
                                        boards_evt.append(int(tokens[0]))
                                        channels_evt.append(int(tokens[1]))
                                        LGs_evt.append(int(tokens[2]))
                                        HGs_evt.append(int(tokens[3]))

                boards.append(boards_evt)
                channels.append(channels_evt)
                LGs.append(LGs_evt)
                HGs.append(HGs_evt)
                                
        return  tstamps, trgids, boards, channels, LGs, HGs

       


def flatten(tstamps, trgids, boards, channels, LGs, HGs):
        tstamps_flat = []
        trgids_flat = []
        
        boards_flat = []
        channels_flat = []
        LGs_flat = []
        HGs_flat = []

        for i in range(len(tstamps)):
                for j in range(len(channels[i])):
                        tstamps_flat.append(tstamps[i])
                        trgids_flat.append(trgids[i])

                        boards_flat.append(boards[i][j])
                        channels_flat.append(channels[i][j])
                        LGs_flat.append(LGs[i][j])
                        HGs_flat.append(HGs[i][j])

        return np.array(tstamps_flat), np.array(trgids_flat), np.array(boards_flat), np.array(channels_flat), np.array(LGs_flat), np.array(HGs_flat)

def save_as_numpy(tstamps, trgids, boards, channels, LGs, HGs, file_out):
        tstamps_flat, trgids_flat, boards_flat, channels_flat, LGs_flat, HGs_flat = flatten(tstamps, trgids, boards, channels, LGs, HGs)

        data = np.array([tstamps_flat, trgids_flat, boards_flat, channels_flat, LGs_flat, HGs_flat])
        np.save(file_out, data)


def save_as_root(tstamps, trgids, boards, channels, LGs, HGs, file_out):
    file = uproot.recreate(file_out)

    file["tree"] = {"t_stamp": tstamps, 
                    "trgid": trgids,
                    "board": boards,
                    "channel": channels,
                    "LG": LGs,
                    "HG": HGs}
    file.close()

def save_as_dataframe(tstamps, trgids, boards, channels, LGs, HGs, file_out):
        tstamps_flat, trgids_flat, boards_flat, channels_flat, LGs_flat, HGs_flat = flatten(tstamps, trgids, boards, channels, LGs, HGs)

        df = pd.DataFrame({'t_stamp': tstamps_flat,
                   'trgid': trgids_flat,
                   'board': boards_flat,
                   'channel': channels_flat,
                   'LG': LGs_flat,
                   'HG': HGs_flat})

        df.to_pickle(file_out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', default=False, action='store_true', help='output as root file')
    parser.add_argument('-p', '--pandas', default=False, action='store_true', help='output as pickled pandas dataframe')
    parser.add_argument('-n', '--numpy',  default=False, action='store_true', help='output as numpy file')
    parser.add_argument('file', type=str, nargs='+', help='Input file(s)')
    
    args = parser.parse_args()

    if not (args.root or args.pandas or args.numpy):
        parser.error('No output format specified, add -r, -p or -n')
    
    if not args.file:
        parser.error('No input file specified')

    for f in args.file:
        if not f.endswith('.txt'):
            parser.error('Input file must be a .txt file')
        
    for f in args.file:
        print(f'{f}: ', end = '')
        tstamps, trgids, boards, channels, LGs, HGs = dt5202_text(f)
        if args.root:
            file_out_root = f.replace('.txt', '.root')
            save_as_root(tstamps, trgids, boards, channels, LGs, HGs, file_out_root)
            print(f'root ', end = '')    
        if args.pandas:
            file_out_pkl = f.replace('.txt', '.pkl')
            save_as_dataframe(tstamps, trgids, boards, channels, LGs, HGs, file_out_pkl)
            print(f'pandas ', end = '')    
        if args.numpy:
            file_out_npy = f.replace('.txt', '.npy')
            save_as_numpy(tstamps, trgids, boards, channels, LGs, HGs, file_out_npy)
            print(f'numpy ', end = '')    

    print('done.')


