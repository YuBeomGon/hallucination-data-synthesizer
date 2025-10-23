# Alignment Pipeline (Step 01)

WhisperX로 Zeroth 오디오와 텍스트를 정렬해 `data/labels/<split>/raw_alignment.jsonl`을 생성하는 절차입니다.

## 입력/출력 경로
- 입력 오디오: `assets/zeroth/<split>/<split>_XXXXX.wav`
- 입력 메타: `data/zeroth/raw_samples_<split>.jsonl`
- 출력 정렬: `data/labels/<split>/raw_alignment.jsonl`

## 설정 (`configs/default_config.yaml`)
```yaml
paths:
  input_audio_dir: "./assets/zeroth"
  raw_samples_path: "./data/zeroth/raw_samples_train.jsonl"
  alignment_output_dir: "./data/labels"

aligner:
  model_name: "large-v3"
  language: "ko"
  device: "cuda"        # GPU 없으면 "cpu"
  compute_type: "float16"  # CPU일 경우 "float32"
  batch_size: 8
  vad_backend: "none"
  diarize: false
  rng_seed: 42
```

## 실행 명령
```bash
conda activate hallucination_synth
bash scripts/pipeline/run_alignment_cpu.sh train --limit 20

# 테스트셋 정렬
bash scripts/pipeline/run_alignment_cpu.sh test --limit 20
```
- 스크립트는 `data/zeroth/raw_samples_<split>.jsonl`을 자동으로 참조하고 `data/labels/<split>/raw_alignment.jsonl`에 결과를 저장합니다.
- 직접 실행할 경우 `--raw-samples`, `--out` 옵션으로 경로를 지정할 수 있습니다.

## 출력 스키마 요약
```json
{
  "sample_id": "85aeffabd848c1d3",
  "audio_path": "assets/zeroth/train/train_000000.wav",
  "text": "...",
  "alignment": {
    "words": [{"w": "안녕하세요", "start": 0.5, "end": 1.21, "conf": 0.95}],
    "tokens": [...],
    "coverage": {
      "speech_coverage": 0.86,
      "aligned_word_ratio": 1.0,
      "avg_conf": 0.96,
      "audio_duration_sec": 12.4
    }
  },
  "speech_regions": [{"start": 0.48, "end": 1.25}],
  "tool_version": {"whisperx": "3.7.4"},
  "model_name": "large-v3",
  "status": "ok",
  "error_msg": null
}
```
- 실패 시 `status: "error"`, `error_msg`에 예외 메시지 저장.
- 자동 전사 텍스트는 `auto_transcript` 필드에 포함되며, 정답 텍스트와 비교할 때 사용합니다.

## GPU/CPU 주의사항
- GPU 미사용 시 `device: "cpu"`, `compute_type: "float32"`로 설정하십시오.
- `vad_backend: "silero"`는 현재 WhisperX API에서 직접 주입이 지원되지 않으므로 `"none"`으로 두는 것을 권장합니다.
- diarization이 필요 없으면 `diarize: false` (pyannote 종속성 로드 방지).

## 검증 체크리스트
- `data/labels/<split>/raw_alignment.jsonl` 라인 수가 입력 샘플 수와 일치하는지 확인
- 몇 개 레코드에서 `alignment.coverage.aligned_word_ratio`가 0.8 이상인지 확인 (낮으면 재실행/필터 고려)
- `status="error"` 레코드가 있는지 확인하고, 필요 시 재시도 또는 샘플 제외
