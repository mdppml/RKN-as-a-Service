#!/usr/bin/env bash
deltas=(3 5 11 25 51 101)
s="8,1000,1000,1000,1000,1000,1000,1000,1000"
nrepetition=1
for (( iter=0; iter<$nrepetition; iter++))
do
	for i in ${deltas[@]};
	do
		./../cmake-build-debug/helper "127.0.0.1" 7777 >> results/auroc_with_tie/auroc_with_tie_delta_helper &
		sleep 1
		./../cmake-build-debug/proxy_auroc_tie 0 9999 "127.0.0.1" 7777 "127.0.0.1" $i $s >> results/auroc_with_tie/auroc_with_tie_delta_p1 &
		sleep 1
		./../cmake-build-debug/proxy_auroc_tie 1 9999 "127.0.0.1" 7777 "127.0.0.1" $i $s >> results/auroc_with_tie/auroc_with_tie_delta_p2 &
		wait 
	done
done
