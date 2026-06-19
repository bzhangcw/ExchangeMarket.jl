#!/usr/bin/env zsh

maxiter=1000
maxtime=500
for n in 10 20 50 100; do
  for m in 1000 5000; do
    # for tp in ces ges; do
    for tp in ces; do
    cmd="julia --project=. --threads=5 revealed/run_test.jl -m $m -n $n -k 300 --k-test 50 --sample-size 50 -t $tp --wealth-function 0 --classes ces -T $maxtime -I $maxiter --tol-obj 0.0 --tol-delta 0.0 --tol-rc 0.0 -v 1  --ces-rho-low 0.1 --interval-logging 1 --method cg &> $m-$n-$tp.log"
    echo $cmd
    eval $cmd
    done
  done
done
