# kaflix-part-model

## script 설명
---
configs/train_config.yaml

- train, evaluate 때 사용되는 설정파일
```yaml
# 설명

### 스파클링 소다에서 사용하는 경로를 받는 변수
sodaflow:

  ### experiment 실행 시 선택한 dataset의 경로를 받는 변수
  ### 다중 dataset 선택 가능
  dataset_path: 
    - "/datasets/kaflix-part-dataset"
    - "/datasets/twincar-part-kaflix/"

  ### experiment 실행 시 선택한 weight의 경로를 받는 변수
  start_ckpt_path: "/artifacts/twincar-part-kaflix/"
  
### 모델 학습시 지정하는 하이퍼 파라미터
train_inputs:
  ...

### 모델 학습 시 weight 파일이 저장되는 경로
MODEL_DIR: "./logs"

### 모델 학습 시 log 파일이 저장되는 경로
LOGS_DIR: "./logs"

### evaluate 진행 시 eval 결과 파일이 저장되는 경로
EVAL_SAVE_PATH: "./eval_results"
```

configs/infer_config.json

- inference 시에 사용되는 설정 파일

train.sh

```bash
# 사용법
$ sh train.sh
```

- 모델 학습진행 시 사용되는 스크립트.
- 사용되는 설정은 configs/train_config.yaml에서 설정 가능

evaluate.py

```bash
# 사용법
$ python evaluate.py 
```

- 학습중이거나, 학습이 완료된 모델 파일을 이용하여 성능 측정을 진행하는 스크립트.
- 사용되는 설정은 configs/train_config.yaml에서 설정 가능

inference.sh

```bash
#  사용법
$ sh inference.sh
```

- 학습이 완료된 모델 파일을 이용하여 모델 탐색 결과 이미지와 결과 json을 생성하는 스크립트.

- configs/infer_config.json을 별도로 사용함.

