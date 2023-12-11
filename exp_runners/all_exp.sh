#!/usr/bin/env bash

echo "Auroc no tie num samples"
./auroc_no_tie_num_samples_exp.sh
echo "Auroc with tie num samples"
./auroc_with_tie_num_samples_exp.sh
echo "Aupr num samples"
./aupr_num_samples_exp.sh

echo "Auroc no tie num parties"
./auroc_no_tie_num_parties_exp.sh
echo "Auroc with tie num parties"
./auroc_with_tie_num_parties_exp.sh
echo "Aupr num parties"
./aupr_num_parties_exp.sh

echo "Auroc no tie delta"
./auroc_no_tie_delta_exp.sh
echo "Auroc with tie delta"
./auroc_with_tie_delta_exp.sh
echo "Aupr delta"
./aupr_delta_exp.sh

echo "Auroc no tie unbalanced"
./auroc_no_tie_unbalanced_exp.sh
echo "Auroc with tie unbalanced"
./auroc_with_tie_unbalanced_exp.sh
echo "Aupr unbalanced"
./aupr_unbalanced_exp.sh
