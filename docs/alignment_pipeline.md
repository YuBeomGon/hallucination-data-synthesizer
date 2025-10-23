# Alignment Pipeline (Step 01)

Zeroth 정답 텍스트를 WhisperX의 CTC aligner에 직접 맞춰 `data/labels/<split>/raw_alignment.jsonl`을 생성합니다. Whisper 추론(ASR 디코딩)은 수행하지 않으며, tokenizer와 align 모델만 사용합니다.

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
  rng_seed: 42
```

## 실행 명령
```bash
conda activate hallucination_synth
bash scripts/pipeline/run_alignment.sh train --limit 20
bash scripts/pipeline/run_alignment.sh test --limit 20
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
    "words": [
      {"w": "안녕하세요", "start": 0.50, "end": 1.21, "score": 0.95}
    ],
    "tokens": [],
    "coverage": {
      "speech_coverage": 1.0,
      "aligned_word_ratio": 1.0,
      "avg_conf": 0.95,
      "audio_duration_sec": 11.9
    }
  },
  "speech_regions": [{"start": 0.50, "end": 11.90}],
  "tool_version": {"whisperx": "3.x"},
  "model_name": "large-v3",
  "status": "ok",
  "error_msg": null
}
```
- 실패 시 `status: "error"`, `error_msg`에 예외 메시지를 저장합니다.
- 자동 전사는 수행하지 않으므로 `auto_transcript`는 정답 텍스트를 그대로 담습니다.

## GPU/CPU 주의사항
- GPU가 없어도 사용 가능하며 `device`를 `cpu`로 설정하면 됩니다.
- Whisper 추론이 없으므로 `batch_size`, `compute_type`, `vad_backend`, `diarize` 설정은 사용되지 않습니다.

## 검증 체크리스트
- `data/labels/<split>/raw_alignment.jsonl` 라인 수가 입력 샘플 수와 일치하는지 확인
- 몇 개 레코드에서 `alignment.words`의 타임스탬프가 실제 오디오와 잘 맞는지 수동 검증
- `status="error"` 레코드가 있는 경우 오류 메시지를 확인하고 필요 시 재시도하세요.
