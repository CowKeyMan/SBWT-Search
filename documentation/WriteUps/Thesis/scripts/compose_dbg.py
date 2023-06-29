#!/bin/python3

"""
A script to extract kmers from a string of raw reads and outputs the kmers as
well as a De Bruijn Graph construction in the mermaid format
"""

import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input-file',
    help='The file with a separate read on each line',
    required=True
)
parser.add_argument(
    '-k', '--kmer-size',
    help='The size k of each kmer',
    type=int,
    required=True
)
parser.add_argument(
    '-r', '--include-reverse-complements',
    help='Include reverse complement data',
    action='store_true',
    required=False,
)
parser.add_argument(
    '-d', '--include-dollars',
    help='Include the dummy nodes as well',
    action='store_true',
    required=False,
)
parser.add_argument(
    '-s', '--print-sorted-kmers',
    help='Print sorted kmers',
    action='store_true',
    required=False,
)
parser.add_argument(
    '-e', '--print-edges',
    help='Print edges as a separate list for the sorted kmers',
    action='store_true',
    required=False,
)
parser.add_argument(
    '-b', '--print-bitvectors',
    help='Print bit vectors for A, C, G and T',
    action='store_true',
    required=False,
)
parser.add_argument(
    '-c', '--print-collapsed',
    help='Print collapsed edges and bitvectors',
    action='store_true',
    required=False,
)
parser.add_argument(
    '-a', '--print-all',
    help='Set all arguments to true except for reverse complements',
    action='store_true',
    required=False,
)
args = vars(parser.parse_args())

if args['print_all']:
    args['include_dollars'] = True
    args['print_sorted_kmers'] = True
    args['print_edges'] = True
    args['print_bitvectors'] = True
    args['print_collapsed'] = True


def reverse_complement(s: str) -> str:
    reverses = {
        'A': 'T',
        'a': 'T',
        'T': 'A',
        't': 'A',
        'C': 'G',
        'c': 'G',
        'G': 'C',
        'g': 'C',
    }
    new_string = []
    for c in s:
        new_string.append(reverses[c])
    return ''.join(new_string[::-1])  # reverse the string


def sort_colex(kmers: set[str]) -> list[str]:
    return sorted(kmers, key=lambda x: x[::-1])


k = args['kmer_size']

with open(args['input_file'], 'r', encoding='utf-8') as f:
    raw_reads = [line.strip() for line in f.readlines()]

rc_reads = [reverse_complement(read) for read in raw_reads]

kmers = []
rc_kmers = []

for line, rc_line in zip(raw_reads, rc_reads):
    if len(line) < k:
        continue
    for i in range(len(line) - (k - 1)):
        kmers.append(line[i: i + k])
        if args['include_reverse_complements']:
            rc_kmers.append(rc_line[i: i + k])


print("Printing kmers:")
for kmer in kmers:
    print(kmer)
print()
print("Printing reverse complements:")
for kmer in rc_kmers:
    print(kmer)
print()

kmers = set(kmers) | set(rc_kmers)

sub_to_pre_and_post = defaultdict(lambda: [set(), set()])

for kmer in kmers:
    sub_to_pre_and_post[kmer[1:]][0].add(kmer)
    sub_to_pre_and_post[kmer[:-1]][1].add(kmer)

if args['include_dollars']:
    sub_to_pre_and_post_adds = defaultdict(lambda: (set(), set()))
    for _, (parents, children) in sub_to_pre_and_post.items():
        if len(parents) == 0:
            for child in children:
                for i in range(1, k + 1):
                    kmer = '$' * (i) + child[:-i]
                    kmers.add(kmer)
                    sub_to_pre_and_post_adds[kmer[1:]][0].add(kmer)
                    sub_to_pre_and_post_adds[kmer[:-1]][1].add(kmer)

    for sub, (parents, children) in sub_to_pre_and_post_adds.items():
        if sub in sub_to_pre_and_post.keys():
            sub_to_pre_and_post[sub][0] = (
                sub_to_pre_and_post[sub][0].union(parents)
            )
            sub_to_pre_and_post[sub][1] = (
                sub_to_pre_and_post[sub][1].union(children)
            )
        else:
            sub_to_pre_and_post[sub] = (parents, children)

if args["print_sorted_kmers"]:
    print("Printing sorted kmers:")
    for kmer in sort_colex(kmers):
        print(kmer)
    print()
    print("Printing sorted reverse complements:")
    for kmer in sort_colex(rc_kmers):
        print(kmer)
    print()


if args['print_edges']:
    print("Printing edges:")
    for kmer in sort_colex(kmers):
        for child in sub_to_pre_and_post[kmer[1:]][1]:
            if kmer != child:
                print(child[-1], end='')
        print()
    print()

if args['print_edges'] and args['print_collapsed']:
    print("Printing collapsed edges:")
    last_kmer_postfix = None
    for kmer in sort_colex(kmers):
        if last_kmer_postfix == kmer[1:]:
            print('\u2205')
            continue
        last_kmer_postfix = kmer[1:]
        for child in sub_to_pre_and_post[kmer[1:]][1]:
            if kmer != child:
                print(child[-1], end='')
        print()
    print()

if args['print_bitvectors']:
    print("Printing bitvectors:")
    print('A C G T')
    for kmer in sort_colex(kmers):
        bitvector = dict(zip('ACGT', ['0'] * 4))
        for child in sub_to_pre_and_post[kmer[1:]][1]:
            if kmer != child:
                bitvector[child[-1]] = '1'
        print(' '.join([bitvector[x] for x in 'ACGT']))
    print()


if args['print_bitvectors'] and args['print_collapsed']:
    c_map = dict(zip('ACGT', [0] * 4))
    print("Printing collapsed bitvectors:")
    print('A C G T')
    last_kmer_postfix = None
    for kmer in sort_colex(kmers):
        bitvector = dict(zip('ACGT', ['0'] * 4))
        if last_kmer_postfix == kmer[1:]:
            print(' '.join([bitvector[x] for x in 'ACGT']))
            continue
        last_kmer_postfix = kmer[1:]
        for child in sub_to_pre_and_post[kmer[1:]][1]:
            if kmer != child:
                bitvector[child[-1]] = '1'
                c_map[child[-1]] += 1
        print(' '.join([bitvector[x] for x in 'ACGT']))
    print()
    print("Printing c_map")
    c_map = [1] + [c_map[x] for x in 'ACGT']
    print(c_map)
    c_map = [sum(c_map[:i + 1]) for i, _ in enumerate(c_map)]
    print(c_map)
    print('\n'.join([str(x) for x in c_map]))

print("Printing graph:")
print()
print("flowchart LR")
for kmer in kmers:
    print(f'  {kmer}(({kmer}))')
for _, (parents, children) in sub_to_pre_and_post.items():
    for parent in parents:
        for child in children:
            if parent != child:
                print(f'  {parent} --{child[-1]}--> {child}')
print()
print(
    'classDef edgeLabel color:white, background-color:#666, text-align:center;'
)
print('classDef nodeLabel font-weight: bold, font-size: 14px;')
