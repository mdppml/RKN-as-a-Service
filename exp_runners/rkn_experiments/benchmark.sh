#!/bin/bash
# {1..8}
network=wan

for funcid in {1..4}
do
	for expind in {1..5} 
	do
		./../../cmake-build-debug/helper "127.0.0.1" 7777 &
		sleep 3
		./../../cmake-build-debug/proxy_test 0 9999 "127.0.0.1" 7777 "127.0.0.1" $funcid 1 $expind $network &
		sleep 3
		./../../cmake-build-debug/proxy_test 1 9999 "127.0.0.1" 7777 "127.0.0.1" $funcid 1 $expind $network
		wait $p1 $p2 $p3
		sleep 3
	done
done

