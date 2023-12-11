dim=20
lmb=0.5
sig=0.4
tfid="a.101.1"
pool="gmp"
enc="one_hot"
# f here does not change the FRAC variable in constant.h. One has to make sure that they are matching for consistency.
f=20
eps="_eps"
network="wan"
flag=1
for id in 1 2 3 4 5
do
	for anc in 8
	do
		for kmer in 4 8 16 32 64 128
		do
			for len in 128 
			do
				./pprkn_synthetic_test.sh $flag $anc $kmer $lmb $sig $id $network $len
				sleep 1
			done
		done
	done
done
