#!/bin/bash
echo $1 $2 $3 $4 $5 $6 $7 $8 # random_flag - anchor_points - k-mer - lambda - sigma - run id - network type - sequence length
frac=20

./../../build/helper "127.0.0.1" 7777 &
p1=$!
sleep 2
./../../build/proxy_rkn 0 9999 "127.0.0.1" 7777 "127.0.0.1" $1 $2 $3 $4 $5 $6 $7 $8 & # > output/no_gt/$1/p0_q$1_k$3_ind$4_f${frac}.out &
p2=$!
sleep 2
./../../build/proxy_rkn 1 9999 "127.0.0.1" 7777 "127.0.0.1" $1 $2 $3 $4 $5 $6 $7 $8  # > output/no_gt/$1/p1_q$1_k$3_ind$4_f${frac}.out
p3=$!
wait $p1 $p2 $p3
