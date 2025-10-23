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

## 2. Zeroth_v2 WAV/JSONL 생성
```bash
python scripts/datasets/export_zeroth_v2.py \
  --dataset kresnik/zeroth_korean \
  --audio-dir assets/zeroth_v2 \
  --raw-samples-dir data/zeroth_v2
```

## 3. 두 발화 합성 (Step 01)
```bash
python -m src.pipeline.step_01_two_utterances \
  --config configs/two_utterances.yaml \
  --split train
python -m src.pipeline.step_01_two_utterances \
  --config configs/two_utterances.yaml \
  --split test
```

## 4. 라벨 생성 (Step 02, 구현 예정)
```bash
# TODO: step_03_build_labels.py 업데이트 후 명령 추가
```

### 결과 확인
```bash
wc -l data/labels_v2/train/paired_meta.jsonl
jq 'select(.status!="ok")' data/labels_v2/train/paired_meta.jsonl | head
```

`docs/two_utterances_pipeline.md`를 참고해 전이 구간 품질을 점검하고, 필요 시 `notebooks/augment_qa.ipynb`에서 오디오를 시각화하세요.
