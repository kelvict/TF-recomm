#!/usr/bin/env bash
for batch_size in 500; do
	for trainset_rate in 0.8 0.2 0.7 0.3 0.6 0.4 0.5; do
		for lr in 0.0001; do
			for dim in 100;do #100>75
				for reg in 0.03; do
					for seed in 0 1 2 3 4; do
					python svd_train_val.py ${seed} ${trainset_rate} ${lr} ${reg} ${dim} ${batch_size}> log/biasMf_${batch_size}_${seed}_${trainset_rate}_${lr}_${reg}_${dim}.log 2>&1 &
					done
				done
			done
		done
	done
done
