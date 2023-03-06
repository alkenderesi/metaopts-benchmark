#! /bin/bash


models=("mp" "dnn" "cnn")
algorithms=("sgd" "adam" "ga" "avoa" "mvo" "dgo" "stbo")
result_dir="results"

for model in "${models[@]}"
do
    mkdir -p "$result_dir/$model"
    for algorithm in "${algorithms[@]}"
    do
        python3 benchmark.py $model $algorithm
        mv *fitness.csv *weights.pickle "$result_dir/$model"
    done
done
