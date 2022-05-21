CUDA_VISIBLE_DEVICES='2,3' python pl_pipeline.py \
    --exp_name=pl_pipline_debug \
    --dataset=fashioniq \
    --dataset_path="/home/lishi/workspace/fashion_iq/start_kit/data/" \
    --savedir=./experiments/fashioniq/ \
    --batch_size 48 \
    --num_workers 32