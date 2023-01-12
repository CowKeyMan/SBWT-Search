#!/bin/python3

"""
Output graphs of batches vs time taken for the files in the log file
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Input log file to analyse', required=True)
parser.add_argument(
    '-t',
    help='Include timings which include copying the SBWT index to GPU',
    action='store_true',
    default=False,
)
args = vars(parser.parse_args())

# read all lines from file
with open(args['i'], 'r', encoding="utf-8") as f:
    lines = f.readlines()


# get dataframe
benchmarks = defaultdict(list)
start = 0
for line in lines:
    if line.startswith('Now running'):
        s = line.split()
        benchmarks['name'].append(s[3])
        benchmarks['batches'].append(int(s[5]))
        benchmarks['format'].append(s[8])
        benchmarks['valid'].append(True)
    else:
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            j['log']['type'] == 'timed_event'
            and j['log']['component'] == ('main' if args['t'] else 'Querier')
        ):
            if j['log']['state'] == 'start':
                start = j['time']
            else:
                end = j['time']
                benchmarks['time'].append(
                    (pd.Timestamp(end) - pd.Timestamp(start))
                    .total_seconds() * 1000
                )
        elif (
            'too large' in j['log']['message']
            or 'cannot be opened' in j['log']['message']
        ):
            benchmarks['valid'][-1] = False
df_full = pd.DataFrame(benchmarks)

# Plot time vs batches
for form in df_full['format'].unique():
    fig, axs = plt.subplots(
        np.ceil(len(df_full['name'].unique()) / 2).astype(int), 2
    )
    for i, (name, ax) in enumerate(zip(df_full['name'].unique(), axs.flat)):
        fig.suptitle(f'Time vs Batches for {form} output format')
        ax.set_title(name, fontsize=8)
        ax.set_xlabel('time (ms)', fontsize=8)
        ax.set_ylabel('batches', fontsize=8)
        ax.set_xticks(
            np.arange(0, df_full['batches'].max() + 1, 100)
        )
        ax.tick_params(axis='both', labelsize=6)
        # ax.set_yticks(ax.get_yticks(), fontsize=5)
        ax.set_xlim(0, df_full['batches'].max())
        df = df_full[
            (df_full['format'] == form)
            & (df_full['name'] == name)
            & df_full['valid']
        ]
        ax.plot(df['batches'], df['time'])
        ax.set_ylim(0, ax.get_ylim()[1])
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig(
        f'{str(Path(args["i"]).parent)}/'
        f'{form}_timings.svg', format='svg', dpi=20000
    )
    plt.close(fig)

for name in df_full['name'].unique():
    fig, ax = plt.subplots()
    df = df_full[
        (df_full['name'] == name)
        & df_full['valid']
    ]
    g = sns.barplot(
        data=df,
        x='format',
        y='time',
        hue='batches',
        ax=ax
    )
    ax.set_ylabel('time (ms)')
    plt.title(f'Time vs batches grouped by format for {name}')
    title = (
        f'{str(Path(args["i"]).parent)}/'
        + 'time_vs_batches_'
        + str(Path(name).name).replace('/', '_').replace('.', '_')
        + '.svg'
    )
    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[f'{x.get_height() / 1000:.0f}s' for x in container],
            fontsize=3
        )
    plt.savefig(title, bbox_inches='tight', format='svg')
    plt.close(fig)
