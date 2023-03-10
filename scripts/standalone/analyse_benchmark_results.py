#!/bin/python3

"""
Produces timeline graphs of when each component is active alongside a table
containing other statistics calculated from the given log file
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Input log file to analyse', required=True)
parser.add_argument(
    '-p', help='Generate Plots of first p values. Set to some large number to '
    'generate all plots',
    required=False,
    default=0,
    type=int
)
parser.add_argument(
    '-f',
    help='Filter only this file and ignore other input files',
    required=False,
    default=None,
    type=str
)
parser.add_argument(
    '-s',
    help='show plots',
    required=False,
    default=False,
    action='store_true'
)
args = vars(parser.parse_args())

query_components = set((
    'SequenceFileParser PositionsBuilder '
    'SeqToBitsConverter Searcher ResultsPrinter'
).split())


def timed_event_to_df_entry(js: dict) -> dict:
    return {
        'time': js['time'],
        'process': js['process'],
        'thread': js['thread'],
        'state': js['log']['state'],
        'component': js['log']['component'],
        'section': js['log']['message'],
    }


# read all lines from file
with open(args['i'], 'r', encoding="utf-8") as f:
    lines = f.readlines()

# separate individual benchmarks
benchmark_name_to_lines = defaultdict(list)
for line in lines:
    line = line.strip()
    if line.startswith('Time taken to copy and build in LOCAL_SCRATCH'):
        continue
    start_string = 'Now running: '
    if line.startswith(start_string):
        current_benchmark_name = line[len(start_string):]
        current_filename = line.split()[3]
    elif not args['f'] or current_filename == args['f']:
        benchmark_name_to_lines[current_benchmark_name].append(line)

# extract useful information from benchmark logs
benchmark_to_details = defaultdict(dict)
for name, lines in benchmark_name_to_lines.items():
    details = benchmark_to_details[name]
    details['timed_event'] = []
    details['num_strings'] = 0
    details['num_chars'] = 0
    details['num_batches'] = 0
    details['num_queries'] = 0
    details['gpu_time'] = 0
    for line in lines:
        try:
            j = json.loads(line)
            if j['log']['type'] == 'message':
                message = j['log']['message']
                if message.startswith('Free gpu memory'):
                    details['gpu_memory'] = (
                        int(message.split()[3])
                    )
                    details['gpu_characters'] = (
                        int(message.split()[9])
                    )
                elif message.startswith('Free main memory'):
                    details['cpu_memory'] = (
                        int(message.split()[3])
                    )
                    details['cpu_characters'] = (
                        int(message.split()[9])
                    )
                elif message.startswith('Using') and 'per batch' in message:
                    details['chars_per_batch'] = (
                        int(message.split()[1])
                    )
                elif message.startswith('Read'):
                    details['num_chars'] += (
                        int(message.split()[1])
                    )
                    details['num_strings'] += (
                        int(message.split()[4])
                    )
                    details['num_batches'] += 1
                elif (
                    message.startswith('Batch ')
                    and ' ms to search in the GPU' in message
                ):
                    details['gpu_time'] += float(message.split()[3])
                elif message.startswith('Batch ') and 'consists of' in message:
                    details['num_queries'] += int(message.split()[4])
            elif j['log']['type'] == 'timed_event':
                details['timed_event'].append(
                    timed_event_to_df_entry(j)
                )
        except json.JSONDecodeError:
            pass


def bits_to_gb(bits):
    return bits / 8 / (1024 ** 3)


# Create dataframe of details with total time included in
for name, details in benchmark_to_details.items():
    df = pd.DataFrame(details['timed_event'])
    df['time'] = pd.to_datetime(df['time'])
    details['df'] = df
    details['total_time'] = (
        (df['time'].max() - df['time'].min()) * 1000
    ).total_seconds()


# Sort the keys of the dataframe by lowest time (best) first
sorted_keys = sorted(
    benchmark_to_details.keys(),
    key=lambda x: benchmark_to_details[x]['total_time']
)
for index, name in enumerate(sorted_keys):
    if index < args['p']:
        fig, axs = plt.subplots(2, 1)
        fig.suptitle(name)
    details = benchmark_to_details[name]
    title = name
    gpu_bits = details['gpu_memory']
    gpu_characters = details['gpu_characters']
    cpu_bits = details['cpu_memory']
    cpu_characters = details['cpu_characters']
    print(f'{index + 1}. Results for {name}:')
    start_text = (
        f'Processed {details["num_chars"]} characters from '
        f'{details["num_strings"]} strings in '
        f'{details["num_batches"]} batches\n'
        f'Total queries: {details["num_queries"]}\n'
        f'Total time taken: {details["total_time"]:.2f}ms\n'
        f'Gpu memory available: {gpu_bits} '
        f'bits = {bits_to_gb(gpu_bits):.2f}GB\t({gpu_characters} characters)\n'
        f'Cpu memory available: {cpu_bits} '
        f'bits = {bits_to_gb(cpu_bits):.2f}GB\t({cpu_characters} characters)\n'
        f'Used {details["chars_per_batch"]} characters per batch. (limited by '
        f'{"cpu" if cpu_characters < gpu_characters else "gpu"} memory)'
    )
    print(start_text)
    df = details['df']
    unique_components = df['component'].unique()[::-1]
    min_time = df['time'].min()
    details['summary'] = {}
    total_query_components = 0
    for i, component in enumerate(unique_components):
        timings = []
        df_component = df[df['component'] == component]
        for section in df[df['component'] == component]['section'].unique():
            df_filtered = df_component[df_component['section'] == section]
            start = (
                df_filtered[df_filtered['state'] == 'start']['time'].iloc[0]
            )
            end = df_filtered[df_filtered['state'] == 'stop']['time'].iloc[0]
            total = (end - start) / np.timedelta64(1, 's') * 1000
            timings.append(total)
            if index < args['p']:
                axs[0].barh(
                    i,
                    total,
                    left=(start - min_time).total_seconds() * 1000,
                    height=0.3
                )
        if component in query_components:
            total_query_components += np.sum(timings)
        details['summary'][component] = {
            'min': np.min(timings),
            'max': np.max(timings),
            'avg': np.average(timings),
            'stdev': np.std(timings),
            'median': np.median(timings),
            'total': np.sum(timings),
        }
    df_summary = pd.DataFrame.from_dict(
        details['summary'], orient="index"
    )
    df_summary = df_summary.round(2)
    print(df_summary.reindex(index=df_summary.index[::-1]))
    actual_query_components = df_summary.loc['Querier', 'total']
    time_saved_query_components = (
        total_query_components - actual_query_components
    )
    nanoseconds_per_query = (
        df_summary.loc['Querier']['total'] * 1000000 / details['num_queries']
    )
    nanoseconds_per_query_gpu = (
        details['gpu_time'] * 1000000 / details['num_queries']
    )
    nanoseconds_per_query_cpu = (
        details['summary']['SearcherSearch']['total']
        * 1000000
        / details['num_queries']
    )
    nanoseconds_per_query_with_memory = (
        details['summary']['Searcher']['total']
        * 1000000
        / details['num_queries']
    )
    end_text = (
        'Total time taken by query components: '
        f'{total_query_components:.2f}ms\n'
        'Actual time taken by query components: '
        f'{actual_query_components:.2f}ms\n'
        '\tThis means that the actual search speed is '
        f'{nanoseconds_per_query} ns/query\n'
        'Time saved by multithreading when querying: '
        f'{time_saved_query_components:.2f}ms\n'
        'Gpu searching according to Event Timers '
        f'time took {details["gpu_time"]}ms\n'
        '\tThis means that the search speed is '
        f'{nanoseconds_per_query_gpu} ns/query\n'
        'Gpu searching according to CPU times '
        f'time took {details["summary"]["SearcherSearch"]["total"]}ms\n'
        '\tThis means that the search speed is '
        f'{nanoseconds_per_query_cpu} ns/query\n'
        'Considering memory transfers as well, we have '
        f'{nanoseconds_per_query_with_memory} ns/query'
    )
    print(end_text)
    print()
    # Plot and save figure
    if index < args['p']:
        table = axs[1].table(
            df_summary.values,
            rowLabels=df_summary.index,
            colLabels=df_summary.columns,
            loc='center'
        )
        table.set_fontsize(6)
        axs[1].axis('off')
        axs[1].axis('tight')
        axs[0].yaxis.set_ticks(
            range(len(unique_components)), unique_components, fontsize=5
        )
        axs[0].set_xlabel('time (ms)')
        axs[0].tick_params(axis='x', labelsize=8)
        plt.figtext(
            0.5,
            -0.40,  # change this if you have more text
            (start_text + '\n' + end_text).replace('\t', ' '),
            ha="center",
            fontsize=8
        )
        input_filename = name.split()[1]
        filename = (
            str(Path(args['i']).parent)
            + "/"
            + input_filename.replace('/', '_').replace('.', '_')
            + "_"
            + name.split()[3]
            + 'batches'
            + '_'
            + name.split()[6]
        )
        filename += '.svg'
        fig.tight_layout()
        plt.savefig(filename, bbox_inches='tight', format='svg', dpi=1200)
        if args['s']:
            plt.show()
