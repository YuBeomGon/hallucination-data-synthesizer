# Noise Dataset Preparation

이 문서는 AI Hub 소음 데이터를 다운로드하고 파이프라인에서 활용할 수 있는 형식으로 정리하는 절차를 설명합니다.

## 1. 다운로드
- 스크립트: `scripts/noise/download_aihub_noises.sh`
- 입력: AI Hub `datasetkey`, `resource_id` (기본값은 도시 소리 데이터 585)
- 출력: `assets/noises/` 아래에 압축이 풀린 WAV/JSON 파일

```bash
bash scripts/noise/download_aihub_noises.sh
# 또는
bash scripts/noise/download_aihub_noises.sh 71376 <RESOURCE_ID>
```

## 2. 카탈로그 생성
- 스크립트: `scripts/noise/build_noise_catalog.py`
- 입력: `assets/noises`
- 출력: `data/noise/noise_catalog.csv`

```bash
python scripts/noise/build_noise_catalog.py \
  --root assets/noises \
  --output data/noise/noise_catalog.csv
```

CSV에는 `clip_start_sec`, `clip_end_sec`, `clip_duration_sec`, 카테고리, dB 값, 원본 오디오 경로가 기록됩니다. 증강 단계에서 노이즈를 무작위 샘플링할 때 참조합니다.

## 3. (선택) 리샘플링 캐시
- 스크립트: `scripts/noise/preprocess_noise_resample.py`
- 입력: `data/noise/noise_catalog.csv`
- 출력: `data/noise/resampled/` 하위에 16 kHz mono WAV, `data/noise/noise_catalog_resampled.csv`

```bash
python scripts/noise/preprocess_noise_resample.py \
  --catalog data/noise/noise_catalog.csv \
  --target-dir data/noise/resampled \
  --output data/noise/noise_catalog_resampled.csv \
  --target-sr 16000 --mono
```

NumPy/SciPy 버전 충돌이 발생하면 `pip install --upgrade scipy` 또는 `pip install "numpy<2"`로 맞춰주세요. 스크립트는 실패한 항목을 경고로 남기고 기존 레코드를 유지합니다.

## 4. 카탈로그 탐색
- 노트북: `notebooks/noise_catalog_overview.ipynb`
- 입력: `data/noise/noise_catalog.csv`
- 목적: 카테고리 분포, 소음 길이 히스토그램, 샘플 청취 등

```bash
conda activate hallucination_synth
jupyter notebook notebooks/noise_catalog_overview.ipynb
```

시각화 결과를 기반으로 증강 시 사용할 카테고리, 길이 범위를 결정하세요.
