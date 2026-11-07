from dataclasses import dataclass
from statistics import mean


@dataclass
class ResultEntry:
    dist: float
    time: float


import sys
sys.stdout = open("tables/baseline.txt", "w")

data: dict[str, list[tuple[ResultEntry | None, ResultEntry | None]]] = {}
for name in ["lcld", "url", "ctu", "wids"]:
    with open(f"raw_results/baseline/{name}.txt") as f:
        entries = [e.split() for e in f.read().splitlines() if not e.startswith("#")]
    data[name] = []
    for i in range(0, len(entries), 2):
        ours = entries[i]
        assert entries[i][0] == "Final-SMT"
        moeva = entries[i + 1]
        assert entries[i + 1][0] == "Final-Moeva"

        data[name].append(
            (
                None
                if len(ours) == 2
                else ResultEntry(dist=float(ours[1]), time=float(ours[2])),
                None
                if len(moeva) == 2
                else ResultEntry(dist=float(moeva[1]), time=float(moeva[2])),
            ),
        )

# table1
print("table 1 --- 攻撃成功率 (左から: 提案手法, Moeva2)")
for name, entries in data.items():
    cnt_success_ours = 0
    cnt_success_moeva = 0
    for ours, moeva in entries:
        if ours is not None:
            cnt_success_ours += 1
        if moeva is not None:
            cnt_success_moeva += 1
    print(name, (cnt_success_ours / len(entries)), (cnt_success_moeva / len(entries)))

# table2
print("table 2 --- 成功の組み合わせ確率 (T/Fは順に 提案手法, Moeva2 を表し、[TT, TF, FT, FF] の順)")
for name, entries in data.items():
    cnt = [0, 0, 0, 0]
    for ours, moeva in entries:
        if ours is not None and moeva is not None:
            cnt[0] += 1
        elif ours is not None and moeva is None:
            cnt[1] += 1
        elif ours is None and moeva is not None:
            cnt[2] += 1
        else:
            cnt[3] += 1
    print(name, [e / len(entries) for e in cnt])


# table3
print("table 3 --- 両手法成功時の平均実行時間 (左から: 提案手法, Moeva2)")
for name, entries in data.items():
    cnt_ours = 0
    sum_ours = 0
    cnt_moeva = 0
    sum_moeva = 0
    for ours, moeva in entries:
        if (ours == None) != (moeva is None):
            continue
        if ours is not None:
            cnt_ours += 1
            sum_ours += ours.time
        if moeva is not None:
            cnt_moeva += 1
            sum_moeva += moeva.time

    print(name, sum_ours / cnt_ours, sum_moeva / cnt_moeva)


# table4
print("table 4 --- 提案手法のみ成功時の平均実行時間 (提案手法)")
for name, entries in data.items():
    ours_time = []
    for ours, moeva in entries:
        if not (ours is not None and moeva is None):
            continue
        ours_time.append(ours.time)

    print(name, mean(ours_time))

# table

print("table 5 --- 両手法成功時の平均距離 (左から: 提案手法, Moeva2)")
for name, entries in data.items():
    ours_dist = []
    moeva_dist = []

    for ours, moeva in entries:
        if (ours is None) or (moeva is None):
            continue
        ours_dist.append(ours.dist)
        moeva_dist.append(moeva.dist)

    print(name, mean(ours_dist), mean(moeva_dist))

print("table 6 --- 提案手法のみ成功時の平均距離 (提案手法)")
for name, entries in data.items():
    ours_dist = []

    for ours, moeva in entries:
        if not (ours is not None and moeva is None):
            continue
        ours_dist.append(ours.dist)

    print(name, mean(ours_dist))
