lmb=0.5
sig=0.4
network="wan"
flag=1
# experiment id
for id in 1 2 3 4 5
do
  # number of anchor points
	for anc in 8
	do
	  # length of kmers
		for kmer in 4 8 16 32 64 128
		do
		  # length of sequence
			for len in 128 
			do
				./pprkn_synthetic_test.sh $flag $anc $kmer $lmb $sig $id $network $len
				sleep 1
			done
		done
	done
done
