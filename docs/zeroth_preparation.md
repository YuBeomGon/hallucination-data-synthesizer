# Zeroth Dataset Preparation

WhisperX 정렬에 사용하기 위해 Hugging Face `Bingsu/zeroth-korean` 데이터를 로컬 WAV와 JSONL 메타로 변환하는 방법을 설명합니다.

## 요구 사항
- `datasets`, `soundfile` 패키지 (이미 `requirements.txt`에 포함)
- Hugging Face 토큰이 필요한 경우 `huggingface-cli login`

## 1. WAV 및 메타데이터 생성
- 스크립트: `scripts/datasets/export_zeroth_raw_samples.py`
- 출력:
  - `assets/zeroth/<split>/<split>_XXXXX.wav`
  - `data/zeroth/raw_samples_<split>.jsonl`

```bash
conda activate hallucination_synth
python scripts/datasets/export_zeroth_raw_samples.py \
  --split train \
  --limit 100 \
  --audio-dir assets/zeroth \
  --output data/zeroth/raw_samples_train.jsonl

python scripts/datasets/export_zeroth_raw_samples.py \
  --split test \
  --audio-dir assets/zeroth \
  --output data/zeroth/raw_samples_test.jsonl
```

> 참고: Hugging Face 캐시에서 직접 파일을 읽기 때문에 최초 실행 시 디코딩 패키지가 요구될 수 있습니다. NumPy/Pandas 버전 충돌이 발생하면 `pip install --upgrade pandas` 또는 `pip install "numpy<2"`로 호환성을 맞춰주세요.

## 2. JSONL 스키마
```json
{
  "sample_id": "85aeffabd848c1d3",
  "audio_path": "train/train_000000.wav",
  "text": "...",
  "language": "ko",
  "split": "train",
  "sr_hz": 16000,
  "channels": 1,
  "dataset": "Bingsu/zeroth-korean",
  "index": 0
}
```
- `sample_id`: `sha1(relative_audio_path + text)`로 결정성 유지
- `audio_path`: `paths.input_audio_dir` 기준 상대 경로
- `text`: Zeroth 제공 전사 (정답)
- `split`: train/test 구분

## 3. 검증 팁
- `data/zeroth/raw_samples_train.jsonl`의 레코드 수와 `assets/zeroth/train` WAV 파일 수가 일치하는지 확인
- WhisperX 정렬 전에 `head`로 몇 개 레코드를 확인해 `audio_path`, `text`가 올바른지 검증
- 추후 SFT/DPO 레이블을 만들 때 정답 텍스트와 자동 전사를 비교할 수 있도록 `text` 필드를 유지하세요.
