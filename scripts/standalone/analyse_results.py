import pandas as pd
import argparse
import json
from matplotlib import pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Input log file to analyse', required=True)
args = vars(parser.parse_args())

with open(args['i'], 'r') as f:
    lines = f.readlines()


def line_to_df_entry(line) -> dict:
    try:
        j = json.loads(line)
    except Exception:
        return None
    if not j['log']['type'] == 'timed_event':
        return None
    return {
        'time': j['time'],
        'process': j['process'],
        'thread': j['thread'],
        'state': j['log']['state'],
        'component': j['log']['component'],
        'section': j['log']['message'],
    }


dicts = [
    line_to_df_entry(line)
    for line in lines
    if line_to_df_entry(line) is not None
]


df = pd.DataFrame(dicts)
df['time'] = pd.to_datetime(df['time'])

# plot here
fig, ax = plt.subplots()
total_time = (df['time'].max() - df['time'].min()).total_seconds()
min_time = df['time'].min()
ax.set_xlim(0, total_time)

unique_components = df['component'].unique()[::-1]
for i, component in enumerate(unique_components):
    for section in df[df['component'] == component]['section'].unique():
        filtered_df = df[
            (df['component'] == component)
            & (df['section'] == section)
        ]
        print(component, section)
        start = filtered_df[filtered_df['state'] == 'start']['time'].iloc[0]
        end = filtered_df[filtered_df['state'] == 'stop']['time'].iloc[0]
        ax.barh(
            i,
            (end - start) / np.timedelta64(1, 's'),
            left=(start - min_time).total_seconds()
        )
ax.yaxis.set_ticks(range(len(unique_components)), unique_components)

plt.show()
