
python baselines/cifar/variational_inference.py \
    --data_dir=/home/thlarsen/.local/lib/python3.6/site-packages/tensorflow_datasets \
    --output_dir=/tmp/vi_model_2 \
    --use_gpu=True \
    --num_cores=1 \
    --download_data=True \
    --train_epochs=2 \
    --num_eval_samples=5 

