# Hallucination Data Synthesizer

Hallucination Data Synthesizer는 OpenAI Whisper 계열과 같은 대규모 자동 음성 인식(ASR) 모델에서 발생하는 환각(hallucination) 문제를 완화하기 위한 학습 데이터를 자동으로 합성하는 파이프라인입니다. 침묵 구간에서의 의미 없는 텍스트 생성을 억제하기 위해 오디오를 프로그래밍 방식으로 증강하고, DPO(Direct Preference Optimization)와 SFT(Supervised Fine-Tuning)에 최적화된 레이블을 생성합니다.

## 핵심 기능
- **정밀한 타임스탬프 정렬**: WhisperX를 사용해 단어/토큰 단위의 고정밀 타임스탬프 추출.
- **프로그래밍 기반 오디오 증강**: 단어 간 간격에 사용자 정의 침묵 또는 배경 소음을 삽입하고 경계면을 스무딩.
- **듀얼 포맷 레이블 생성**: DPO용 chosen/rejected 쌍과 SFT용 `<SIL>` 토큰 포함 타깃 스크립트, 재계산 타임스탬프 생성.
- **설정 기반 파이프라인**: 모든 하이퍼파라미터와 경로를 YAML 한 곳에서 관리.
- **완전한 재현성 보장**: 증강 이력과 모델 정보를 메타데이터로 저장하여 추적 가능성 확보.

## 프로젝트 구조
```text
hallucination-data-synthesizer/
├── README.md
├── docs/
│   ├── noise_preparation.md
│   ├── zeroth_preparation.md
│   └── alignment_pipeline.md
├── requirements.txt
├── configs/
│   └── default_config.yaml
├── scripts/
│   ├── install.sh
│   ├── datasets/
│   │   └── export_zeroth_raw_samples.py
│   ├── noise/
│   │   ├── download_aihub_noises.sh
│   │   ├── build_noise_catalog.py
│   │   └── preprocess_noise_resample.py
│   └── pipeline/
│       ├── run_alignment_cpu.sh
│       └── run_synthesis.sh
├── assets/
│   ├── noises/
│   └── zeroth/
├── data/
│   ├── augmented_audio/
│   │   ├── train/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   │   └── raw_alignment.jsonl
│   │   └── test/
│   ├── noise/
│   │   ├── noise_catalog.csv
│   │   └── resampled/
│   └── zeroth/
│       ├── raw_samples_train.jsonl
│       └── raw_samples_test.jsonl
└── src/
    ├── main.py
    ├── pipeline/
    │   ├── step_01_align.py
    │   ├── step_02_augment.py
    │   └── step_03_build_labels.py
    ├── modules/
    │   ├── audio_processor.py
    │   ├── label_generator.py
    │   └── whisperx_wrapper.py
    └── utils/
        ├── file_io.py
        ├── logging_config.py
        └── config_loader.py
```

## 설치
사전 요구 사항:
- Python 3.10 이상
- `ffmpeg` (시스템 전역 설치)

설치 절차:
```bash
git clone https://github.com/your-username/hallucination-data-synthesizer.git
cd hallucination-data-synthesizer

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

bash scripts/install.sh
```

## 설정
`configs/default_config.yaml` 파일에서 데이터 경로, 증강 옵션, 모델 이름 등을 관리합니다. 예시:

```yaml
paths:
  input_audio_dir: "/path/to/your/original/dataset"
  noise_dir: "./assets/noises"
  output_dir: "./output/generated_dataset"
  noise_catalog: "./data/noise/noise_catalog.csv"
  noise_resampled_dir: "./data/noise/resampled"
  raw_samples_path: "./data/zeroth/raw_samples_train.jsonl"
  alignment_output_dir: "./data/labels"

aligner:
  model_name: "large-v3"
  language: "ko"
  device: "cuda"
  compute_type: "float16"   # CPU 사용 시 "float32" 권장
  batch_size: 8
  vad_backend: "none"
  diarize: false
  rng_seed: 42

synthesis:
  min_gap_ms: 1000
  insertion_duration_ms: 3000
  insertion_type: "noise"  # "silence" 또는 "noise"
  crossfade_ms: 50

labelling:
  baseline_model_name: "openai/whisper-large-v3"
```

## POC 데이터 자원
- **음성(Prompt) 데이터**: Hugging Face의 `Bingsu/zeroth-korean`을 활용해 초기 실험을 진행할 수 있습니다. `datasets` 라이브러리로 바로 로드할 수 있으며, `audio`, `text` 필드를 포함합니다.
  ```python
  from datasets import load_dataset

  dataset = load_dataset("Bingsu/zeroth-korean")
  first_sample = dataset["train"][0]
  ```
  데이터셋은 16 kHz 샘플링 레이트의 음성과 한국어 전사를 제공합니다.
- **소음(Augmentation) 데이터**: AI Hub에서 배포하는 아래 두 가지 소음 데이터셋을 사용해 침묵/소음 삽입 효과를 테스트할 수 있습니다.
  - 극한 소음 환경 소리 데이터: <https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71376>
  - 도시 소리 데이터: <https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=585>
  라이선스와 이용 약관을 확인하고 프로젝트 목적에 맞게 사용 경로를 `configs/default_config.yaml`에 지정하세요.

## Noise 데이터 준비 절차
1. **다운로드**  
   `scripts/noise/download_aihub_noises.sh`를 실행해 AI Hub 데이터를 `assets/noises/`에 추출합니다.
   ```bash
   bash scripts/noise/download_aihub_noises.sh
   # 예시: bash scripts/noise/download_aihub_noises.sh 585 4C228107-8608-482B-AC25-E2E91F17E122
   ```
2. **카탈로그 생성**  
   ```bash
   python scripts/noise/build_noise_catalog.py \
     --root assets/noises \
     --output data/noise/noise_catalog.csv
   ```
3. **(선택) 리샘플링 캐시 생성**  
   ```bash
   python scripts/noise/preprocess_noise_resample.py \
     --catalog data/noise/noise_catalog.csv \
     --target-dir data/noise/resampled \
     --output data/noise/noise_catalog_resampled.csv \
     --target-sr 16000 --mono
   ```
4. **카탈로그 탐색**  
   `notebooks/noise_catalog_overview.ipynb`를 열어 카테고리 분포와 길이를 분석합니다.

자세한 단계별 설명은 `docs/noise_preparation.md`를 참고하세요.

## 추가 문서
- `docs/zeroth_preparation.md`: Zeroth 데이터 추출 및 JSONL 작성 방법
- `docs/alignment_pipeline.md`: WhisperX 정렬 파이프라인 실행 및 출력 검증

## 사용 방법
1. `configs/default_config.yaml`을 프로젝트 환경에 맞게 수정합니다.
2. 세부 단계는 `docs/` 폴더 문서를 참고해 순차적으로 실행합니다.
3. 전체 파이프라인을 실행합니다.
   ```bash
   bash scripts/pipeline/run_synthesis.sh
   ```
4. 결과물은 `output_dir`에 지정한 경로에 생성되며, 증강된 오디오와 `metadata.jsonl` 파일이 포함됩니다.

## 파이프라인 처리 원칙
- **ID 결정성**: `sample_id = sha1(relative_audio_path + text)`, 증강 후 `aug_id = f"{sample_id}_{hash(augment_events)}`처럼 항상 같은 입력에 동일 식별자를 부여합니다.
- **시간 단위**: 전 구간에서 초 단위(float)만 사용하며 밀리초 표기는 금지합니다.
- **샘플레이트/채널 유지**: 기본 16 kHz mono를 유지하고 변경 시 `resample_info` 메타에 기록합니다.
- **버전/시드 추적**: 모든 산출물에 `tool_version`, `model_name`, `rng_seed`를 포함합니다.
- **실패 내구성**: 각 레코드에 `status(ok/skip/error)`와 `error_msg`를 남겨 재처리 대상 식별이 용이하도록 합니다.

## 파이프라인 단계별 I/O 계약
### Step 01 – Alignment (`src/pipeline/step_01_align.py`)
입력:
- Zeroth 등 원본 데이터를 전처리한 `data/zeroth/raw_samples_<split>.jsonl`
- WhisperX 정렬 설정(`aligner.model_name`, `device`, `language`, `batch_size`, `vad_backend`, `diarize` 등)

출력(`data/labels/<split>/raw_alignment.jsonl`):
```json
{
  "sample_id": "zeroth_train_000123",
  "audio_path": "assets/zeroth/train/train_000123.wav",
  "text": "안녕하세요 반갑습니다.",
  "alignment": {
    "words": [
      {"w": "안녕하세요", "start": 0.52, "end": 1.21, "conf": 0.95},
      {"w": "반갑습니다", "start": 1.82, "end": 2.49, "conf": 0.98}
    ],
    "tokens": [
      {"t": "안", "start": 0.52, "end": 0.62},
      {"t": "녕", "start": 0.62, "end": 0.69}
    ],
    "coverage": {
      "speech_coverage": 0.86,
      "aligned_word_ratio": 1.00,
      "avg_conf": 0.965
    }
  },
  "speech_regions": [
    {"start": 0.48, "end": 1.25},
    {"start": 1.78, "end": 2.55}
  ],
  "tool_version": {"whisperx": "x.y.z"},
  "model_name": "large-v3",
  "rng_seed": 42,
  "status": "ok",
  "error_msg": null
}
```
체크포인트:
- 정렬 실패 시에도 레코드를 남기고 `status="error"`로 표기합니다.
- 문장부호·공백 정규화 규칙을 사전 정의해 WER/정렬 비교의 일관성을 유지합니다.
- 30초 이상 긴 오디오는 WhisperX 세그 단위 정렬 후 `segment_boundaries`로 경계를 저장합니다.

### Step 02 – Augmentation (`src/pipeline/step_02_augment.py`)
입력:
- `data/labels/<split>/raw_alignment.jsonl`
- 증강 설정(`insertion_type`, `min_gap_ms`, `insertion_duration_ms`, `crossfade_ms`, `snr_db`, `loudness_target_lufs`, `limit_true_peak_dbfs` 등)
- 다운로드된 소음 파일(`assets/noises`) 또는 침묵 삽입 옵션

출력:
- 증강 오디오 `data/augmented_audio/<split>/{aug_id}.wav`
- 증강 메타 `data/labels/<split>/augmented_meta.jsonl`
```json
{
  "aug_id": "zeroth_train_000123_abcd12",
  "sample_id": "zeroth_train_000123",
  "original_audio_path": "assets/zeroth/train/train_000123.wav",
  "augmented_audio_path": "data/augmented_audio/train/zeroth_train_000123_abcd12.wav",
  "augmentation": {
    "events": [
      {
        "type": "insert_silence",
        "start_orig": 1.25,
        "duration": 3.00,
        "snr_db": null,
        "crossfade_ms": 50,
        "noise_src": null
      }
    ],
    "postprocess": {
      "loudness_target_lufs": -23.0,
      "true_peak_dbfs": -1.0,
      "clip_guard_applied": true
    }
  },
  "offset_map": [
    {"t0_src": 0.00, "t0_dst": 0.00},
    {"t0_src": 1.25, "t0_dst": 1.25},
    {"t0_src": 1.25, "t0_dst": 4.25, "delta": 3.00}
  ],
  "updated_segments": [
    {"w": "안녕하세요", "start": 0.52, "end": 1.21},
    {"w": "반갑습니다", "start": 4.82, "end": 5.49}
  ],
  "tool_version": {"ffmpeg": "...", "librosa": "..."},
  "rng_seed": 4242,
  "status": "ok",
  "error_msg": null
}
```
권장 사항:
- 빠른 경로: `offset_map` 기반으로 기존 정렬을 이동.
- 정밀 경로(옵션): 증강 오디오를 WhisperX로 재정렬, `updated_segments_refined`에 보존.
- 증강 전/후 LUFS·True Peak 측정치 기록, 삽입 이벤트 결정은 고정된 RNG 시드로 재현성을 확보합니다.

### Step 03 – Label Build (`src/pipeline/step_03_build_labels.py`)
입력:
- `augmented_meta.jsonl`
- 베이스라인 STT 추론 결과 (또는 모듈 내에서 직접 추론)
  - 보수적 디코딩: `condition_on_previous_text=False`, `temperature=0`, `beam_size=5–8`
  - 유도 디코딩: `temperature↑`, `beam_size=1`, `condition_on_previous_text=True`
- 각 디코딩의 내부 지표(`avg_logprob`, `compression_ratio`, `no_speech_prob`) 저장

출력(`data/labels/metadata.jsonl`):
```json
{
  "aug_id": "zeroth_train_000123_abcd12",
  "audio_path": "data/augmented_audio/zeroth_train_000123_abcd12.wav",
  "dpo": {
    "chosen": {
      "text": "안녕하세요. <SIL> 반갑습니다.",
      "decode_params": {"temp": 0.0, "beam": 6, "cond_prev": false},
      "metrics": {"avg_logprob": -0.12, "compression_ratio": 1.12, "no_speech_prob": 0.03}
    },
    "rejected": {
      "text": "안녕하세요. 감사합니다. 감사합니다. 반갑습니다.",
      "decode_params": {"temp": 0.8, "beam": 1, "cond_prev": true},
      "metrics": {"avg_logprob": -1.05, "compression_ratio": 2.85, "no_speech_prob": 0.01}
    },
    "mask": {
      "type": "insert_alignment",
      "spans": [{"start_tok": 3, "end_tok": 7}],
      "confidence": 0.91
    }
  },
  "sft": {
    "target_text": "안녕하세요. <SIL> 반갑습니다.",
    "silences_meta": [{"start": 1.25, "end": 4.25}],
    "label_masking": "only_sil",
    "special_tokens": ["<SIL>"]
  },
  "eval": {
    "reference_text": "안녕하세요. 반갑습니다.",
    "wer_chosen": 0.00,
    "wer_rejected": 0.36,
    "ir_chosen": 0.00,
    "ir_rejected": 0.28,
    "dr_chosen": 0.00,
    "dr_rejected": 0.03,
    "her_proxy": 0.28
  },
  "meta": {
    "original_audio_path": "data/original/train/000123.wav",
    "dataset": "zeroth_korean",
    "split": "train",
    "augmentation": {
      "type": "silence",
      "start_sec": 1.25,
      "duration_sec": 3.00,
      "crossfade_ms": 50
    },
    "aligner_model": "whisperx-large-v3",
    "stt_model": "openai/whisper-tiny",
    "tool_version": {"synth": "0.1.0"},
    "rng_seed": 4242
  },
  "status": "ok",
  "error_msg": null
}
```
핵심 포인트:
- Masked-DPO 지원을 위해 `dpo.mask`에 토큰 스팬을 명시합니다.
- 정답이 있으면 정답 정렬 기반으로 삽입 위치를 계산하고, 없으면 정렬·품질 지표에서 불일치 토큰을 추정합니다.
- `<SIL>`은 텍스트에 1회만 삽입하고 실제 길이는 `silences_meta`로 관리하여 평가 시 `<SIL>`을 제거한 WER 계산을 지원합니다.
- 모든 디코딩 파라미터와 지표를 저장해 재현성과 오류 분석을 확보합니다.

## HF 데이터셋 변환 권장 포맷
- **DPO split**
  ```json
  {
    "audio": {"path": "data/augmented_audio/xxx.wav", "sampling_rate": 16000},
    "chosen": "...",
    "rejected": "...",
    "mask_spans": [[3, 7], [12, 14]],
    "meta": {...}
  }
  ```
- **SFT split**
  ```json
  {
    "audio": {"path": "...", "sampling_rate": 16000},
    "text": "... <SIL> ...",
    "silences_meta": [{"start": 1.25, "end": 4.25}],
    "masking": "only_sil",
    "meta": {...}
  }
  ```

## 에지 케이스 및 방어 로직
- 정렬 불안정(`aligned_word_ratio < 0.8`) 시 `status="skip"`으로 마킹합니다.
- STT 결과의 `compression_ratio > 2.6`이면 환각 가능성으로 플래그하고 라벨 신뢰도를 낮게 기록합니다.
- `<SIL>`은 연속 삽입을 금지하며 최소 길이(≥0.8–1.0초)를 만족하는 구간만 사용합니다.
- 삽입 노이즈는 SNR dB 기준으로 스케일링하고 LUFS/True Peak 제한으로 클리핑을 방지합니다.
- 30초 초과 오디오는 세그먼트 단위로 나눠 증강 후 `offset_map`으로 전체 오프셋을 유지합니다.
- RNG 시드, 선택된 노이즈 파일, 삽입 위치, 디코딩 파라미터를 모두 메타에 남겨 결정성을 확보합니다.

## 출력 데이터 스키마
`output/metadata.jsonl`의 각 라인은 DPO, SFT, 메타정보를 포함한 JSON 구조입니다.

```json
{
  "dpo": {
    "audio_path": "output/augmented_audio/sample_001.wav",
    "chosen": "안녕하세요. 반갑습니다.",
    "rejected": "안녕하세요. 감사합니다. 감사합니다. 시청해주셔서 감사합니다."
  },
  "sft": {
    "audio_path": "output/augmented_audio/sample_001.wav",
    "target_text": "안녕하세요. <SIL> 반갑습니다.",
    "word_segments": [...]
  },
  "meta": {
    "original_audio_path": "path/to/original/sample_001.wav",
    "dataset": "ai_hub_dialogue",
    "split": "train",
    "augmentation": {
      "type": "noise",
      "source": "assets/noises/cafe_noise.wav",
      "start_sec": 1.8,
      "duration_sec": 3.0
    },
    "rejected_model": "openai/whisper-large-v3",
    "original_word_segments": [
      {"word": "안녕하세요", "start": 0.5, "end": 1.2, "score": 0.95},
      {"word": "반갑습니다", "start": 1.8, "end": 2.5, "score": 0.98}
    ]
  }
}
```

주요 필드:
- `dpo`: 증강 오디오를 프롬프트로 사용하여 chosen/rejected 텍스트 쌍을 제공.
- `sft`: `<SIL>` 토큰으로 침묵 구간을 명시한 타깃 텍스트 및 재계산된 단어 타임스탬프.
- `meta`: 데이터 생성 이력을 모두 추적할 수 있는 메타데이터.

## 개발 가이드라인
- **포매터 & 린터**: `black`, `ruff` (또는 `flake8`) 사용.
- **타입 힌트**: 모든 함수와 주요 변수에 명시적 타입 힌트를 작성.
  - **Docstring**: 모듈, 클래스, 함수에 Google 스타일 Docstring 작성.
  - **로깅**: 표준 출력 대신 `logging` 모듈 사용.
  - **설정 관리**: 모든 하드코딩된 설정값은 YAML 구성으로 이전.
  - **모듈화**: 각 모듈은 단일 책임 원칙(SRP)을 준수.
- **검증 리소스**: 자동화 테스트는 `tests/` 폴더에, 실험용 노트북은 `notebooks/`에 배치합니다. `pytest` 도입 후 단계별 I/O 계약을 검증하고 재실행 시 동일 결과가 나오는지 확인하세요.

## 라이선스
프로젝트에 적용할 라이선스가 정해지지 않았다면, 사용 목적에 맞는 라이선스를 추가하세요.

## 기여
이슈와 풀 리퀘스트는 언제나 환영합니다. 버그 리포트나 기능 제안 시 설정 파일, 사용한 모델 버전, 로그를 함께 제공해 주세요.
