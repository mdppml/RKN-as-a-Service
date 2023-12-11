#!/usr/bin/env bash
s="8,12,18,32,58,107,258,507,1008"
nrepetition=1
for (( iter=0; iter<$nrepetition; iter++))
do
	echo " ======== Iteration $iter ========== "
	./../cmake-build-debug/helper "127.0.0.1" 7777 >> results/aupr/aupr_unbalanced_helper &
	sleep 1
	./../cmake-build-debug/proxy_aupr 0 9999 "127.0.0.1" 7777 "127.0.0.1" 1 $s >> results/aupr/aupr_unbalanced_p1 &
	sleep 1
	./../cmake-build-debug/proxy_aupr 1 9999 "127.0.0.1" 7777 "127.0.0.1" 1 $s >> results/aupr/aupr_unbalanced_p2 &
	wait 
done
