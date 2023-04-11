#!/bin/python3

"""
Script to analyse the benchmark output.
"""

import argparse
try:
    import ujson as json
except ImportError:
    import json
import re
import sys
from collections import defaultdict
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

total_time_component = 'Querier'


ordered_components = [
    # 'main',
    # 'SBWTLoader',
    # 'SBWTParserAndIndex',
    'SBWTReadAndPopppy',
    'SbwtGpuTransfer',
    'Presearcher',
    # 'PresearchFunction',
    'MemoryAllocator',
    # 'Querier',
    'IndexFileParser',
    'SequenceFileParser',
    'PositionsBuilder',
    'SeqToBitsConverter',
    'Searcher',
    # 'SearcherCopyToGpu',
    # 'SearcherSearch',
    # 'SearcherCopyFromGpu',
    'ResultsPostProcessor',
    'ResultsPrinter',
]

query_components = {
    'IndexFileParser',
    'SequenceFileParser',
    'PositionsBuilder',
    'SeqToBitsConverter',
    'Searcher',
    'ResultsPostProcessor',
    'ResultsPrinter',
}

tabled_components = {
    'IndexFileParser',
    'SequenceFileParser',
    'PositionsBuilder',
    'SeqToBitsConverter',
    'Searcher',
    'ResultsPostProcessor',
    'ResultsPrinter',
}


parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input',
    help='Input log file to analyse',
    required=True
)
parser.add_argument(
    '-l', '--include_loading',
    help='Include loading time',
    action='store_true',
    required=False
)
parser.add_argument(
    '-m', '--include_memory_allcator',
    help='Include memory allocation time',
    action='store_true',
    required=False
)
args = vars(parser.parse_args())


class DataFrameGenerator:
    def __init__(self):
        self.batch_stream_to_chars_reads = {}
        self.line_pattern_to_parser = {
            r'Time taken to copy and build in LOCAL_SCRATCH: \d* ms':
                self.pass_parser,
            (
                r'Now running: File (.*) with (\d*) streams in '
                r'(.*) format on (.*) device'
            ): self.running_parser,
            r'(^{.*}$)': self.json_parser,
        }
        self.message_pattern_to_parser = {
            r'Loading components into memory': self.pass_parser,
            r'Using kmer size: (\d*)': self.kmer_size_parser,
            r'Using (\d*) streams': self.streams_parser,
            (
                r'Free gpu memory: \d* bits \(\.*\). '
                r'This allows for (\d*) characters per batch'
            ): self.gpu_memory_parser,
            (
                r'Free main memory: \d* bits \(\d*\). '
                r'This allows for (\d*) characters per batch'
            ): self.cpu_memory_parser,
            (
                r'Using (\d*) max characters per batch '
                r'and (\d*) max reads per batch'
            ): self.max_chars_and_reads_parser,
            r'Running OpenMP with (\d*) threads': self.threads_parser,
            r'Running queries': self.pass_parser,
            r'Now reading file .*$': self.pass_parser,
            (
                r'Batch (\d*) stream (\d*) contains '
                r'(\d*) indexes in (\d*) reads'
            ):
                self.batch_size_parser,
        }

    def generate(self, file: str) -> pd.DataFrame:
        with open(file, 'r', encoding="utf-8") as f:
            lines = f.readlines()
        self.dfs = []
        self.working_dict = defaultdict(list)
        closest_percentage = -1
        for line_no, line in enumerate(lines):
            cp = round(line_no / len(lines) * 100)
            if cp != closest_percentage:
                print(f'{cp}% ', end='', flush=True)
                closest_percentage = cp
            try:
                self.parse_line(line.strip())
            except Exception as exc:
                raise Exception(
                    f"Error in line {line_no + 1}: \"{line}\""
                ) from exc
        self.merge_working_dict()
        print()
        df = pd.concat(self.dfs)
        df = df.astype({
            'component': 'category',
            'stream': int,
            'time': float,
            'state': 'category',
            'batch': int,
            'file': 'category',
            'print_mode': 'category',
            'device': 'category',
            'streams_total': int,
        })
        return df

    def parse_line(self, line: str):
        if len(line) == 0:
            return
        for (pattern, parser) in self.line_pattern_to_parser.items():
            match = re.match(pattern, line)
            if match is not None:
                parser(*match.groups())
                return
        raise RuntimeError

    def pass_parser(self):
        pass

    def running_parser(
        self, input_file: str, streams: str, print_mode: str, device: str
    ):
        self.merge_working_dict()
        self.current_input_file = input_file
        self.current_streams = streams
        self.current_print_mode = print_mode
        self.current_device = device

    def json_parser(self, s: list[str]):
        d = json.loads(f'{s}')
        if d['log']['type'] == 'message':
            for (pattern, parser) in self.message_pattern_to_parser.items():
                match = re.match(pattern, d['log']['message'])
                if match is not None:
                    parser(*match.groups())
        elif d['log']['type'] == 'timed_event':
            self.timed_event_parser(d)
        else:
            raise RuntimeError

    def kmer_size_parser(self, kmer_size: int):
        self.kmer_size = kmer_size

    def streams_parser(self, streams: int):
        self.streams = streams

    def gpu_memory_parser(self, memory: str, chars_per_batch: int):
        self.gpu_memory = memory
        self.gpu_max_chars_per_batch = chars_per_batch

    def cpu_memory_parser(self, memory: str, chars_per_batch: int):
        self.cpu_memory = memory
        self.cpu_max_chars_per_batch = chars_per_batch

    def max_chars_and_reads_parser(self, max_chars: int, max_reads: int):
        self.max_chars_per_batch = max_chars
        self.max_reads_per_batch = max_reads

    def threads_parser(self, threads):
        self.threads = threads

    def batch_size_parser(self, batch, stream, chars, reads):
        self.batch_stream_to_chars_reads[(batch, stream)] = (chars, reads)

    def timed_event_parser(self, d: dict):
        if '_' in d['log']['component']:
            self.working_dict['component'].append(
                d['log']['component'].split('_')[0]
            )
            self.working_dict['stream'].append(
                d['log']['component'].split('_')[1]
            )
        else:
            self.working_dict['component'].append(d['log']['component'])
            self.working_dict['stream'].append(0)
        self.working_dict['time'].append(d['time'])
        self.working_dict['state'].append(d['log']['state'])
        self.working_dict['batch'].append(
            self.batch_parser(d['log']['message'])
        )

    def batch_parser(self, message: str) -> int | None:
        match = re.match(r'batch (\d*)', message)
        if match:
            return match.groups()[0]
        return 0

    def merge_working_dict(self):
        if len(self.working_dict) == 0:
            return
        df = pd.DataFrame.from_dict(self.working_dict)
        self.working_dict = defaultdict(list)
        df['file'] = Path(self.current_input_file).stem
        df['print_mode'] = self.current_print_mode
        df['device'] = self.current_device
        df['time'] = pd.to_datetime(df['time'])
        df['time'] = (
            df['time'] - pd.to_datetime(df['time'].min())
        ).dt.total_seconds()
        # + 1 because stream ids start from 0
        df['streams_total'] = df['stream'].dropna().astype(int).max() + 1
        self.dfs.append(df)


def filter_df_by_series(df: pd.DataFrame, s: pd.Series | dict) -> pd.DataFrame:
    for key, value in s.items():
        df = df[df[key] == value]
    return df


def get_time_for_component(df: pd.DataFrame, component: str) -> float:
    df = df[df['component'] == component]
    start_time = df[df['state'] == 'start']['time'].iloc[0]
    stop_time = df[df['state'] == 'stop']['time'].iloc[0]
    return stop_time - start_time


class StreamsVsTime:
    def __init__(
        self,
        df: pd.DataFrame,
        output_folder: str,
        include_loader: bool = True,
        include_memory_allocator: bool = True
    ):
        self.df = df
        self.output_folder = output_folder
        self.include_loader = include_loader
        self.include_memory_allocator = include_memory_allocator

    def generate(self):
        unique_graphs = self.df['device file'.split()].drop_duplicates()
        for _, ug in unique_graphs.iterrows():
            df = filter_df_by_series(self.df, ug)
            self.generate_graph(
                df,
                self.output_folder
                    / f"StreamsVsTime_{ug['file']}_{ug['device']}.svg",
                "Streams vs Time separated by different output formats\n"
                f"for file {ug['file']} running on {ug['device']} hardware"
            )

    def generate_graph(
        self, df: pd.DataFrame, output_filename: str, title: str
    ):
        graph_dict = defaultdict(list)
        unique_graph_cols = (
            self.df['print_mode streams_total'.split()]
        ).drop_duplicates()
        for _, ugc in unique_graph_cols.iterrows():
            try:
                filtered_df = filter_df_by_series(df, ugc)
                graph_dict['print_mode'].append(ugc['print_mode'])
                graph_dict['streams_total'].append(ugc['streams_total'])
                graph_dict['query_time'].append(
                    get_time_for_component(filtered_df, 'Querier')
                )
                graph_dict['memory_alloc_time'].append(
                    get_time_for_component(filtered_df, 'MemoryAllocator')
                )
                if 'SBWTLoader' in filtered_df['component'].values:
                    graph_dict['loading_time'].append(
                        get_time_for_component(filtered_df, 'SBWTLoader')
                    )
                elif 'ColorsLoader' in filtered_df['component'].values:
                    graph_dict['loading_time'].append(
                        get_time_for_component(filtered_df, 'ColorsLoader')
                    )
                else:
                    raise RuntimeError("No Loader found")
            except IndexError:
                print(
                    f"Error at {output_filename}"
                    f", {ugc['print_mode']}, {ugc['streams_total']} streams"
                )
                sys.exit(1)
        df = pd.DataFrame(graph_dict)
        print(title)
        print(df)
        print()
        df['memory_alloc_time'] += df['query_time']
        df['loading_time'] += (
            df['memory_alloc_time'] if self.include_memory_allocator
            else df['query_time']
        )
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.title(title)
        if self.include_loader:
            self.generate_bar_plot(ax, df, 'loading_time', 'Set3', False)
        if self.include_memory_allocator:
            self.generate_bar_plot(ax, df, 'memory_alloc_time', 'Set2', False)
        self.generate_bar_plot(ax, df, 'query_time', 'Set1', True)
        handles, labels = ax.get_legend_handles_labels()
        handles = handles[-len(df['streams_total'].unique()):]
        labels = labels[-len(df['streams_total'].unique()):]
        plt.legend(
            handles,
            labels,
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            title='streams'
        )
        plt.savefig(output_filename, bbox_inches='tight', format='svg')
        plt.close(fig)

    def generate_bar_plot(
        self,
        ax: plt.axis,
        df: pd.DataFrame,
        variable: str,
        palette: str,
        print_labels: str
    ):
        sns.barplot(
            data=df,
            x='print_mode',
            y=variable,
            hue='streams_total',
            palette=palette,
            ax=ax
        )
        ax.set(xlabel='Print Mode', ylabel='Time (s)')
        if not print_labels:
            return
        fontsize = 13
        for container in ax.containers[
            int(-len(df) / len(df['print_mode'].unique())):
        ]:
            ax.bar_label(
                container,
                labels=[f'{x.get_height():.0f}' for x in container],
                fontsize=fontsize,
                padding=-(fontsize + 3)
            )


class IndividualAnalysis:
    def __init__(
        self,
        df: pd.DataFrame,
        output_folder: str,
        include_loader: bool = False,
        include_memory_allocator: bool = False
    ):
        self.df = df
        self.output_folder = output_folder
        self.include_loader = include_loader
        self.include_memory_allocator = include_memory_allocator

    def generate(self):
        unique_graph_cols = (
            self.df['file print_mode device streams_total'.split()]
        ).drop_duplicates()
        ugcs = []
        for _, ugc in unique_graph_cols.iterrows():
            single_df = filter_df_by_series(df, ugc)
            ugcs.append((ugc, get_time_for_component(single_df, 'Querier')))
        unique_graph_cols_sorted = pd.DataFrame([
            x[0] for x in
            sorted(ugcs, key=lambda ugc: ugc[1])
        ])
        for _, ugc in unique_graph_cols_sorted.iterrows():
            single_df = filter_df_by_series(df, ugc)
            self.generate_single(
                single_df,
                ugc['file'],
                ugc['device'],
                ugc['streams_total'],
                ugc['print_mode'],
            )

    def generate_single(
        self,
        df: pd.DataFrame,
        file: str,
        device: str,
        streams_total: str,
        print_mode: str
    ):
        fig, axs = plt.subplots(2, 1)
        plt.subplots_adjust(wspace=0, hspace=0.05)
        title = (
            f"Individual Analysis for file {file} running\n"
            f"on {device} hardware with {streams_total} "
            f"streams and outputting in {print_mode} format"
        )
        axs[0].set_title(title)
        print(title)
        self.generate_timeline(axs[0], df)
        self.generate_table(axs[1], df)
        plt.savefig(
            self.output_folder
            / f"Individual_{file}_{device}_"
            f"{print_mode}_{streams_total}_streams.svg",
            bbox_inches='tight',
            format='svg'
        )
        plt.close(fig)

    def generate_timeline(self, ax: plt.axis, df: pd.DataFrame):
        index = 0
        components_df = df['component stream'.split()].drop_duplicates()
        components_df = components_df.sort_values(
            by='component',
            key=lambda column: [
                ordered_components.index(x) if x in ordered_components
                else -1
                for x in column
            ],
            ascending=False
        )
        height = 0.1
        component_ticks = []
        for _, row in (
            components_df
            .reset_index(drop=True)
            .iterrows()
        ):
            if row['component'] not in ordered_components:
                continue
            component_ticks.append(
                f"{row['component']}_{row['stream']}"
            )
            filtered_df = filter_df_by_series(df, row)
            for batch_df in (
                filtered_df[filtered_df['batch'] == x]
                for x in filtered_df['batch'].unique()
            ):
                height += 0.03
                start_time = (
                    batch_df[batch_df['state'] == 'start']['time'].iloc[0]
                )
                stop_time = (
                    batch_df[batch_df['state'] == 'stop']['time'].iloc[0]
                )
                ax.barh(
                    index,
                    width=stop_time - start_time,
                    height=0.3,
                    left=start_time,
                )
            index += 1
        ax.figure.set_size_inches(15, height)
        ax.yaxis.set_ticks(
            range(len(component_ticks)), component_ticks, fontsize=11
        )
        ax.set_xlabel('Time (s)')

    def generate_table(self, ax: plt.axis, df: pd.DataFrame):
        component_to_batch_time = defaultdict(list)
        component_to_stream_time = defaultdict(
            lambda: [0] * df['streams_total'].iloc[0]
        )
        components_df = df['component stream'.split()].drop_duplicates()
        for _, row in components_df.iterrows():
            filtered_df = filter_df_by_series(df, row)
            component = row['component']
            for batch_df in (
                filtered_df[filtered_df['batch'] == x]
                for x in filtered_df['batch'].unique()
            ):
                time = get_time_for_component(batch_df, component)
                component_to_batch_time[component].append(time)
                component_to_stream_time[row['component']][row['stream']] += (
                    time
                )
        result_df = pd.DataFrame()
        batches_values = component_to_batch_time.values()
        streams_values = component_to_stream_time.values()
        result_df['component'] = list(component_to_batch_time.keys())
        # result_df['total_batches'] = [len(x) for x in batches_values]
        # result_df['avg_batch_time'] = [np.mean(x) for x in batches_values]
        # result_df['median_batch_time'] = [
        #     np.median(x) for x in batches_values
        # ]
        # result_df['std_dev_batch_time'] = [np.std(x) for x in batches_values]
        result_df['min_stream_time'] = [np.min(x) for x in streams_values]
        result_df['max_stream_time'] = [np.max(x) for x in streams_values]
        # result_df['avg_stream_time'] = [np.mean(x) for x in streams_values]
        # result_df['std_dev_stream_time'] = [
        #     np.std(x) for x in streams_values
        # ]
        result_df['total_time'] = [np.sum(x) for x in batches_values]
        print(result_df)
        ax.table(
            result_df.values,
            colLabels=result_df.columns,
            loc='best',
            fontsize=6
        )
        ax.axis('off')
        ax.axis('tight')


input_file = Path(args['input'])
output_folder = input_file.parent

print('Parsing: ', end='')
df = DataFrameGenerator().generate(input_file)

print('Generating streams vs time:')
StreamsVsTime(df, output_folder).generate()

print('Generating IndividualAnalysis:')
IndividualAnalysis(df, output_folder).generate()

# import cProfile
# import pstats
# import sys

# def print_profiling_results(pr) -> None:
#     pstats.Stats(pr, stream=sys.stdout) \
#         .sort_stats(pstats.SortKey.CUMULATIVE) \
#         .print_stats()

# with cProfile.Profile() as pr:
# print_profiling_results(pr)
