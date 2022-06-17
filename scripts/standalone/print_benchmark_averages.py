import sys
from collections import defaultdict

filename = sys.argv[1]

function_to_timings = defaultdict(list)

with open(filename, 'r') as f:
    for line in f:
        if line.strip() == '':
            continue
        function, timing = line.split(':')
        function = function.strip()
        timing = float(timing.strip())
        function_to_timings[function].append(timing)

function_to_average = {
    function: sum(timings) / len(timings)
    for function, timings in function_to_timings.items()
}

for function, average in function_to_average.items():
    print(f'{function}: {average}')
