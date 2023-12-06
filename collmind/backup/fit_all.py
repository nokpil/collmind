from os.path import join
import time
import json
import pickle
import os
import sys
from argparse import ArgumentParser
from typing import List, Tuple
from glob import glob
from collections import defaultdict
import gc
import subprocess

import pandas as pd
import numpy as np
from tqdm import tqdm

collection_name = 'Thehill'
mode = 'fine'
fine_dict = {'Breitbart': (200, 30), 'Gatewaypundit': (400, 30), 'Thehill': (300, 90)}

if mode == 'coarse':
    h = [200, 300, 400]
    u = [30, 60, 90]
else:
    h = fine_dict[collection_name][0]
    u = fine_dict[collection_name][1]
    h = [h-25, h, h+25]
    u = [u-10, u, u+10]

print(h, u)

command_list = [f'nohup python grid_search.py --collection_name {collection_name} --hlist "[{h[0]}]" --ulist "{u}" --slist "[1, 4, 2, 3, 5]" > fit_{mode}_{collection_name}_{h[0]}.out &',
                f'nohup python grid_search.py --collection_name {collection_name}  --hlist "[{h[1]}]" --ulist "{u}" --slist "[2, 5, 1, 3, 4]" > fit_{mode}_{collection_name}_{h[1]}.out &',
                f'nohup python grid_search.py --collection_name {collection_name}  --hlist "[{h[2]}]" --ulist "{u}" --slist "[3, 1, 2, 4, 5]" > fit_{mode}_{collection_name}_{h[2]}.out &'
]

for command in command_list:
    print(f'Current : {command}')
    while True:
        try:
            subprocess.run(command, shell=True, check=True)
            break
        except subprocess.CalledProcessError:
            print(f"Command {command} did not terminate correctly. Re-running...")
            continue