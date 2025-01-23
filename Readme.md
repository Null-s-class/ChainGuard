python run.py \
    --output_dir=saved_models \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=Data/Dataset/data.txt \
    --eval_data_file=Data/Dataset/Valid.txt \
    --test_data_file=Data/Dtaset/Test.txt \
    --epoch 5 \
    --code_length 512 \
    --data_flow_length 128 \
    --train_batch_size 4 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \ 
    --evaluate_during_training \
    --seed 123456 2>&1| tee saved_models/train.log

    

python run.py \
    --output_dir=saved_models \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_eval \
    --do_test \
    --train_data_file=dataset/train.txt \
    --eval_data_file=dataset/valid.txt \
    --test_data_file=dataset/test.txt \
    --epoch 1 \
    --code_length 512 \
    --data_flow_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee saved_models/test.log




