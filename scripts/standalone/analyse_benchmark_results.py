import pandas as pd
import argparse
import json
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Input log file to analyse', required=True)
parser.add_argument(
    '-p', help='Generate Plots', required=False, default=0, type=int
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
parser.add_argument(
    '-r',
    help='remove invalid runs',
    required=False,
    default=False,
    action='store_true'
)
args = vars(parser.parse_args())

query_components = set((
    'SequenceFileParser PositionsBuilder '
    'SeqToBitsConverter Searcher ResultsPrinter'
).split())


def timed_event_to_df_entry(j: dict) -> dict:
    return {
        'time': j['time'],
        'process': j['process'],
        'thread': j['thread'],
        'state': j['log']['state'],
        'component': j['log']['component'],
        'section': j['log']['message'],
    }


# read all lines from file
with open(args['i'], 'r') as f:
    lines = f.readlines()

# separate individual benchmarks
benchmark_name_to_lines = defaultdict(list)
for line in lines:
    line = line.strip()
    start_string = 'Now running: '
    if line.startswith(start_string):
        current_benchmark_name = line[len(start_string):]
        current_filename = line.split()[3]
    else:
        if not args['f'] or current_filename == args['f']:
            benchmark_name_to_lines[current_benchmark_name].append(line)

# extract useful information from benchmark logs
benchmark_to_details = defaultdict(dict)
for name, lines in benchmark_name_to_lines.items():
    details = benchmark_to_details[name]
    details['timed_event'] = []
    details['valid'] = True
    details['num_strings'] = 0
    details['num_chars'] = 0
    details['num_batches'] = 0
    details['num_queries'] = 0
    for line in lines:
        try:
            j = json.loads(line)
            if j['log']['type'] == 'message':
                message = j['log']['message']
                if message.startswith('Free gpu memory'):
                    details['gpu_memory'] = (
                        int(message.split()[3])
                    )
                elif message.startswith('Free main memory'):
                    details['cpu_memory'] = (
                        int(message.split()[3])
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
                elif message.startswith('Batch ') and 'consists of' in message:
                    details['num_queries'] += int(message.split()[4])
                elif (
                    'too large' in j['log']['message']
                    or 'cannot be opened' in j['log']['message']
                ):
                    details['valid'] = False
                    if args['r']:
                        del details
                        break
            elif j['log']['type'] == 'timed_event':
                details['timed_event'].append(
                    timed_event_to_df_entry(j)
                )
        except Exception:
            if line.startswith("Input file size: "):
                details["input_file_size"] = line.split()[3]
            elif line.startswith("Output file size: "):
                details["output_file_size"] = line.split()[3]


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
    valid_text = " (INVALID)" if not details["valid"] else ""
    print(title + valid_text)
    gpu_bits = details['gpu_memory']
    gpu_characters = gpu_bits // 66
    cpu_bits = details['cpu_memory']
    cpu_characters = cpu_bits // 460
    start_text = (
        f'Input file size is {details["input_file_size"]}\n'
        f'Output file size is {details["output_file_size"]}\n'
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
    start_query_components = df[
        [c in query_components for c in df['component']]
    ]['time'].min()
    end_query_components = df[
        [c in query_components for c in df['component']]
    ]['time'].max()
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
    actual_query_components = (
        end_query_components - start_query_components
    ).total_seconds() * 1000
    time_saved_query_components = (
        total_query_components - actual_query_components
    )
    end_text = (
        'Total time taken by query components: '
        f'{total_query_components:.2f}ms\n'
        'Actual time taken by query components: '
        f'{actual_query_components:.2f}ms\n'
        'Time saved by multithreading when querying: '
        f'{time_saved_query_components:.2f}ms'
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
            -0.25,
            (start_text + '\n' + end_text).replace('\t', ' '),
            ha="center",
            fontsize=8
        )
        input_filename = name.split()[1]
        filename = (
            str(Path(args['i']).parent)
            + "/"
            + input_filename.replace('/', '_').replace('.', '_')
            + name.split()[3]
            + 'batches'
            + '_'
            + name.split()[6]
        )
        if not benchmark_to_details[name]['valid']:
            filename += '-invalid'
            props = dict(boxstyle='round', facecolor='red', alpha=0.5)
            axs[0].text(
                0.4,
                1.2,
                'INVALID',
                transform=axs[0].transAxes,
                fontsize=14,
                verticalalignment='top',
                bbox=props
            )
        filename += '.svg'
        fig.tight_layout()
        fig.figsize = (15, 10)
        plt.savefig(filename, bbox_inches='tight', format='svg', dpi=1200)
        if args['s']:
            plt.show()
