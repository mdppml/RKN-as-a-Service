#!/bin/bash

anc=128
kmer=10
lmb=0.5
sig=0.4
network=lan


for id in 1
do
	for ind in {12..189} 
	do
		./../../cmake-build-debug/helper "127.0.0.1" 7777 &
		sleep 3
		./../../cmake-build-debug/proxy_rkn 0 9999 "127.0.0.1" 7777 "127.0.0.1" 0 $anc $kmer $lmb $sig $id $network $ind &
		sleep 3
		./../../cmake-build-debug/proxy_rkn 1 9999 "127.0.0.1" 7777 "127.0.0.1" 0 $anc $kmer $lmb $sig $id $network $ind 
		wait $p1 $p2 $p3
		sleep 3
	done
done

