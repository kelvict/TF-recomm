#!/usr/bin/env bash
for batch_size in 500; do
	for trainset_rate in 0.9; do
		for lr in 0.0002; do
			for dim in 100 75;do #100>75
				for reg in  0.02 0.025 0.0175; do
					for seed in 0; do
					python svd_train_val.py ${seed} ${trainset_rate} ${lr} ${reg} ${dim} ${batch_size}> log/biasMf_${batch_size}_${seed}_${trainset_rate}_${lr}_${reg}_${dim}.log 2>&1 &
					done
				done
			done
		done
	done
done
