import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import subprocess

env = os.environ.copy()
env['LANG'] = 'en_US.UTF-8'
env['LC_ALL'] = 'en_US.UTF-8'

def dessiner_courbe(file_path,file):
    df = pd.read_csv(file_path)
    plt.figure(figsize=(10,5))
    plt.plot(df['Step'], df['Value'])
    plt.xlabel('Steps')
    plt.ylabel(file[:-4])
    match = re.search(r'/([^/]+)/[^/]*$', file_path)
    if match:
        match = match.group(1)
    else:
        return None
    plt.title(match)
    plt.grid()
    # plt.show()
    save_path = 'fig/' + match + '/' 
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    plt.savefig(save_path + file[:-4] + '.pdf', format='pdf')

def list_all_dirs(base_path):
    directories = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file == '.DS_Store': continue
            file_path = os.path.join(root,file)
            dessiner_courbe(file_path,file)
            directories.append(file_path)
    return directories

base_path = 'data/'
files = ['reward.csv','fps.csv','actorloss.csv','criticloss.csv']
all_file_paths = list_all_dirs(base_path)
fig_path = 'fig/'
# print(all_file_paths)
# for file in all_file_paths:
#     dessiner_courbe(file,fig_path)



