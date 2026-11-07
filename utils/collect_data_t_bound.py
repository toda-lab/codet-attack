import glob
import re
from dataclasses import dataclass
from statistics import mean

@dataclass
class ResultEntry:
    dist: float
    time: float

import sys
sys.stdout = open("tables/with_t_bound.txt", "w")

# dict of dataset_name -> dict of index -> [ours_no_t, ours_t, moeva]
raw_data: dict[str, dict[int, list[ResultEntry | None]]] = {}

for filepath in glob.glob("raw_results/with_t_bound/*.txt"):
    with open(filepath) as f:
        lines = f.read().splitlines()
    
    dataset_name = None
    for line in lines:
        if line.startswith("# setting:"):
            m = re.search(r"model_path='dataset/([^/]+)/", line)
            if m:
                dataset_name = m.group(1)
            break
            
    if not dataset_name:
        continue
        
    if dataset_name not in raw_data:
        raw_data[dataset_name] = {}
        
    current_index = None
    
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("# index:"):
            current_index = int(line.split(":")[1].strip())
            if current_index not in raw_data[dataset_name]:
                raw_data[dataset_name][current_index] = [None, None, None]
            idx += 1
            continue
            
        if line.startswith("# use_t_bound: False"):
            idx += 1
            if idx < len(lines):
                result_line = lines[idx]
                parts = result_line.split()
                if len(parts) == 3 and parts[0] == "Final-SMT":
                    raw_data[dataset_name][current_index][0] = ResultEntry(float(parts[1]), float(parts[2]))
            idx += 1
            continue
            
        if line.startswith("# use_t_bound: True"):
            idx += 1
            if idx < len(lines):
                result_line = lines[idx]
                parts = result_line.split()
                if len(parts) == 3 and parts[0] == "Final-SMT":
                    raw_data[dataset_name][current_index][1] = ResultEntry(float(parts[1]), float(parts[2]))
            idx += 1
            continue
            
        if line.startswith("# Moeva2"):
            idx += 1
            if idx < len(lines):
                result_line = lines[idx]
                parts = result_line.split()
                if len(parts) == 3 and parts[0] == "Final-Moeva":
                    raw_data[dataset_name][current_index][2] = ResultEntry(float(parts[1]), float(parts[2]))
            idx += 1
            continue
            
        idx += 1

data: dict[str, list[tuple[ResultEntry | None, ResultEntry | None, ResultEntry | None]]] = {}
for name in ["lcld", "url", "ctu", "wids"]:
    if name not in raw_data:
        continue
    entries = []
    for i in range(100):
        if i in raw_data[name]:
            entries.append(tuple(raw_data[name][i]))
    data[name] = entries

# table1
print("table 1 --- 攻撃成功率 (左から: t_boundなし, t_boundあり, Moeva2)")
for name, entries in data.items():
    cnt_success_ours_no_t = 0
    cnt_success_ours_t = 0
    cnt_success_moeva = 0
    for ours_no_t, ours_t, moeva in entries:
        if ours_no_t is not None:
            cnt_success_ours_no_t += 1
        if ours_t is not None:
            cnt_success_ours_t += 1
        if moeva is not None:
            cnt_success_moeva += 1
    print(name, (cnt_success_ours_no_t / len(entries)), (cnt_success_ours_t / len(entries)), (cnt_success_moeva / len(entries)))

# table2
print("table 2 --- 成功の組み合わせ確率 (T/Fは順に t_boundなし, t_boundあり, Moeva2 を表し、[TTT, TTF, TFT, TFF, FTT, FTF, FFT, FFF] の順)")
for name, entries in data.items():
    cnt = [0] * 8
    for ours_no_t, ours_t, moeva in entries:
        idx = (4 if ours_no_t else 0) + (2 if ours_t else 0) + (1 if moeva else 0)
        cnt[7 - idx] += 1
    print(name, [e / len(entries) for e in cnt])

# table3
print("table 3 --- 全手法成功時の平均実行時間 (左から: t_boundなし, t_boundあり, Moeva2)")
for name, entries in data.items():
    cnt_ours_no_t = 0
    sum_ours_no_t = 0
    cnt_ours_t = 0
    sum_ours_t = 0
    cnt_moeva = 0
    sum_moeva = 0
    for ours_no_t, ours_t, moeva in entries:
        if (ours_no_t is None) or (ours_t is None) or (moeva is None):
            continue
        cnt_ours_no_t += 1
        sum_ours_no_t += ours_no_t.time
        cnt_ours_t += 1
        sum_ours_t += ours_t.time
        cnt_moeva += 1
        sum_moeva += moeva.time

    print(name, sum_ours_no_t / cnt_ours_no_t if cnt_ours_no_t else 0, sum_ours_t / cnt_ours_t if cnt_ours_t else 0, sum_moeva / cnt_moeva if cnt_moeva else 0)

# table4
print("table 4 --- 提案手法のみ成功時の平均実行時間 (左から: t_boundなし, t_boundあり)")
for name, entries in data.items():
    ours_no_t_time = []
    ours_t_time = []
    for ours_no_t, ours_t, moeva in entries:
        if ours_no_t is not None and moeva is None:
            ours_no_t_time.append(ours_no_t.time)
        if ours_t is not None and moeva is None:
            ours_t_time.append(ours_t.time)

    print(name, mean(ours_no_t_time) if ours_no_t_time else 0, mean(ours_t_time) if ours_t_time else 0)

# table
print("table 5 --- 全手法成功時の平均距離 (左から: t_boundなし, t_boundあり, Moeva2)")
for name, entries in data.items():
    ours_no_t_dist = []
    ours_t_dist = []
    moeva_dist = []

    for ours_no_t, ours_t, moeva in entries:
        if (ours_no_t is None) or (ours_t is None) or (moeva is None):
            continue
        ours_no_t_dist.append(ours_no_t.dist)
        ours_t_dist.append(ours_t.dist)
        moeva_dist.append(moeva.dist)

    print(name, mean(ours_no_t_dist) if ours_no_t_dist else 0, mean(ours_t_dist) if ours_t_dist else 0, mean(moeva_dist) if moeva_dist else 0)

print("table 6 --- 提案手法のみ成功時の平均距離 (左から: t_boundなし, t_boundあり)")
for name, entries in data.items():
    ours_no_t_dist = []
    ours_t_dist = []

    for ours_no_t, ours_t, moeva in entries:
        if ours_no_t is not None and moeva is None:
            ours_no_t_dist.append(ours_no_t.dist)
        if ours_t is not None and moeva is None:
            ours_t_dist.append(ours_t.dist)

    print(name, mean(ours_no_t_dist) if ours_no_t_dist else 0, mean(ours_t_dist) if ours_t_dist else 0)
