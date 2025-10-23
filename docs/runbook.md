# Runbook

증강 파이프라인을 실행하기 위한 단계별 명령 모음입니다.

## 1. 노이즈 준비
```bash
bash scripts/noise/download_aihub_noises.sh
python scripts/noise/build_noise_catalog.py \
  --root assets/noises \
  --output data/noise/noise_catalog.csv
# (선택) 리샘플 캐시
python scripts/noise/preprocess_noise_resample.py \
  --catalog data/noise/noise_catalog.csv \
  --target-dir data/noise/resampled \
  --output data/noise/noise_catalog_resampled.csv \
  --target-sr 16000 --mono
```

## 2. Zeroth WAV/JSONL 생성
```bash
python scripts/datasets/export_zeroth_raw_samples.py \
  --split train --audio-dir assets/zeroth \
  --output data/zeroth/raw_samples_train.jsonl
python scripts/datasets/export_zeroth_raw_samples.py \
  --split test --audio-dir assets/zeroth \
  --output data/zeroth/raw_samples_test.jsonl
```

## 3. WhisperX 정렬 (Step 01)
```bash
bash scripts/pipeline/run_alignment.sh train
bash scripts/pipeline/run_alignment.sh test
```

## 4. 증강 (Step 02)
```bash
python -m src.pipeline.step_02_augment \
  --config configs/default_config.yaml \
  --split train
python -m src.pipeline.step_02_augment \
  --config configs/default_config.yaml \
  --split test
```

### 결과 확인
```bash
wc -l data/labels/train/augmented_meta.jsonl
jq 'select(.status=="error")' data/labels/train/augmented_meta.jsonl | head
```

필요시 `notebooks/augment_qa.ipynb`를 열어 품질을 시각적으로 확인하세요.
