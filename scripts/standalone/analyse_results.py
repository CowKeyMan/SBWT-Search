import pandas as pd
import argparse
import re
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Input log file to analyse', required=True)
args = vars(parser.parse_args())

with open(args['i'], 'r') as f:
    lines = f.readlines()

r = re.compile(r"\[(.*?)\] \[(.*?)\] (.*?)$")

matches = []
for line in lines:
    match = r.search(line)
    if match is not None:
        matches.append(list(match.groups()))

df = pd.DataFrame(matches, columns='datetime log_type log'.split())

valid_rows = [
    'has started' in x or 'has finished' in x for x in df['log']
]
df = df[valid_rows]
df['datetime'] = pd.to_datetime(df['datetime'])
df = df[df['log_type'] == 'trace']
df['component'] = [x.split()[0] for x in df['log']]
df['is_start'] = ['has started' in x for x in df['log']]
df['batch_no'] = [x.split()[-1] for x in df['log']]

component_to_timings = defaultdict(list)

for component in df['component'].unique():
    for batch in df['batch_no'].unique():
        df_com_bat = df[
            (df['component'] == component)
            & (df['batch_no'] == batch)
        ]
        start_times = (
            df_com_bat[df_com_bat['is_start']]['datetime'].sort_values()
        )
        end_times = (
            df_com_bat[~df_com_bat['is_start']]['datetime'].sort_values()
        )
        assert len(start_times) == len(end_times)
        for s, e in zip(start_times, end_times):
            millis: pd.Timedelta = (e - s).total_seconds() * 10**3
            component_to_timings[component].append(millis)

print(
    pd.DataFrame(
        [[c, sum(t)] for c, t in component_to_timings.items()],
        columns=['component', 'total_time']
    )
)
