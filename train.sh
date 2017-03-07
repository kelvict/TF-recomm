#!/usr/bin/env bash
for trainset_rate in 0.9; do
    for lr in 0.0005 0.0001; do
		for dim in 50 75 100;do
			for reg in  0.02 0.01; do
				for seed in 0; do
				python svd_train_val.py ${seed} ${trainset_rate} ${lr} ${reg} ${dim} > log/biasMf_${seed}_${trainset_rate}_${lr}_${reg}_${dim}.log 2>&1 &
				done
			done
		done
    done
done
