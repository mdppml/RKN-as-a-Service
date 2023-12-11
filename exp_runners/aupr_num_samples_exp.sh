#!/usr/bin/env bash
nsamples=(64 128 256 512 1024)
nparties=3
nrepetition=1
for (( iter=0; iter<$nrepetition; iter++))
do
	for i in ${nsamples[@]};
	do
		s="$nparties"
		for (( c=0; c<$nparties; c++ ))  
		do
			s="${s},${i}"
		done
		echo " ======= Iteration: $iter - Num of Samples: $i ============ "
		./../cmake-build-debug/helper "127.0.0.1" 7777 >> results/aupr/aupr_num_samples_helper &
		sleep 1
		./../cmake-build-debug/proxy_aupr 0 9999 "127.0.0.1" 7777 "127.0.0.1" 1 $s >> results/aupr/aupr_num_samples_p1 &
		sleep 1
		./../cmake-build-debug/proxy_aupr 1 9999 "127.0.0.1" 7777 "127.0.0.1" 1 $s >> results/aupr/aupr_num_samples_p2 &
		wait 
	done
done
