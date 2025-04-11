<!-- python run.py \
 --output_dir=saved_models \
 --config_name=microsoft/graphcodebert-base \
 --model_name_or_path=microsoft/graphcodebert-base \
 --tokenizer_name=microsoft/graphcodebert-base \
 --do_train \
 --train_data_file=Data/Dataset/datatest.txt \
 --eval_data_file=Data/Dataset/Validtest.txt \
 --test_data_file=Data/Dtaset/Testtest.txt \
 --epoch 1 \
 --code_length 512 \
 --data_flow_length 128 \
 --train_batch_size 8 \
 --eval_batch_size 32 \
 --learning_rate 2e-5 \
 --max_grad_norm 1.0 \
--seed 123456 2>&1| tee saved_models/train.log

--train_data_file=Data/Dataset/data.txt \
--eval_data_file=Data/Dataset/Valid.txt \
--test_data_file=Data/Dtaset/Test.txt \
--epoch 5 \
--evaluate_during_training \

python run.py \
 --output_dir=saved_models \
 --config_name=microsoft/graphcodebert-base \
 --model_name_or_path=microsoft/graphcodebert-base \
 --tokenizer_name=microsoft/graphcodebert-base \
 --do_eval \
 --do_test \
 --train_data_file=Data/Dataset/datatest.txt \
 --eval_data_file=Data/Dataset/Validtest.txt \
 --test_data_file=Data/Dtaset/Testtest.txt \
 --epoch 1 \
 --code_length 512 \
 --data_flow_length 128 \
 --train_batch_size 64 \
 --eval_batch_size 32 \
 --learning_rate 2e-5 \
 --max_grad_norm 1.0 \
 --evaluate_during_training \
 --seed 123456 2>&1| tee saved_models/test.log -->

![Static Badge](https://img.shields.io/badge/Docker_desktop-latest-cyan)
![Static Badge](https://img.shields.io/badge/Python-3.10-blue)
![Static Badge](https://img.shields.io/badge/CUDA-12.1-darkgreen)
 # ChainGuard

## 🚀 Tính năng nổi bật

- **Phân tích đồ thị mã nguồn**: Trích xuất cấu trúc mã thông qua cây cú pháp trừu tượng (AST) và biểu đồ luồng điều khiển (CFG).
- **Tích hợp CodeBERT**: Sử dụng mô hình GraphCodeBERT để hiểu ngữ cảnh mã và phát hiện lỗ hổng tiềm ẩn.
- **Hỗ trợ Docker**: Dễ dàng triển khai và chạy mô hình trong môi trường Docker.

## 📁 Cấu trúc thư mục

- `Data.7z`: Tập dữ liệu huấn luyện và kiểm tra.
- `Processdata.7z`: Dữ liệu đã được xử lý sẵn.
- `parser.7z`: Công cụ phân tích cú pháp mã nguồn.
- `model.py`: Định nghĩa kiến trúc mô hình.
- `run.py`: Tập lệnh huấn luyện và đánh giá mô hình.
- `Tree-sitter.py`: Tập lệnh sử dụng Tree-sitter để phân tích cú pháp mã nguồn.
- `docker-compose.yml`, `Dockerfile`: Tệp cấu hình Docker để triển khai môi trường.

## 🛠️ Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/Null-s-class/ChainGuard.git
cd ChainGuard
```
### 2. Cài đặt các dependency 

```bash
pip install -r requirements.txt
```

### 3. Giải nén dữ liệu

Giải nén các file .7z:

- Data.7z
- Processdata.7z
- parser.7z
Vào các thư mục tương ứng trong project.

### 4. Chạy mô hình

```bash
    python run.py --output_dir=saved_models --config_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base --tokenizer_name=microsoft/graphcodebert-base --do_test --train_data_file=Data/Dataset/data.txt --eval_data_file=Data/Dataset/Valid.txt --test_data_file=Data/Dataset/test.txt --epoch 1 --code_length 512 --data_flow_length 128 --train_batch_size 64 --eval_batch_size 32 --learning_rate 2e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 2>&1| tee saved_models/test.log
```

## 🐳 Sử dụng Docker

### 1. Chạy container

```bash 
docker-compose up --build
```

* Có thể bỏ flag `--build` sau lần chạy đầu tiên. Chi tiết [Docker docs](https://docs.docker.com/compose/)

### 2. Truy cập container

```bash
docker exec -it <id/name container> /bin/bash
```

