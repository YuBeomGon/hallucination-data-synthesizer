# Hallucination Data Synthesizer – Two-Utterance Augmentation

이 브랜치는 동일 화자의 두 개 발화를 이어 붙이고, 연결 구간에 노이즈를 삽입해 ASR 모델의 침묵 환각을 유발·완화하는 학습 데이터를 합성합니다. 기존 단일 발화 중심 파이프라인과 별도로 동작하며, 새 데이터 경로와 설정 파일을 통해 실험을 격리합니다.

## 이 브랜치가 해결하는 문제
- 긴 문맥이나 발화 전환 시점에서 발생하는 환각 사례를 재현하기 위해 두 발화를 한 샘플로 결합합니다.
- 연결 구간에 노이즈·무음을 주입해 모델이 침묵을 과도하게 채우는 패턴을 학습 단계에서 통제합니다.
- 재현 가능한 파이프라인과 메타데이터를 제공해 DPO/SFT 학습, 평가, 오류 분석을 한번에 준비합니다.

## 핵심 기능
- **동화자 발화 페어링**: `speaker_id`를 기준으로 페어 목록을 구성하고 필터링 조건(길이, 간격, 품질)을 설정할 수 있습니다.
- **노이즈 전이 구간 합성**: 발화 A와 B 사이에 프로그램 방식으로 노이즈를 삽입하고 crossfade·LUFS 제어를 적용해 부드러운 전환을 만듭니다.
- **결합 라벨 생성**: 두 발화의 재정렬 결과를 결합해 `<SIL_TRANS>` 등 특수 토큰을 포함한 SFT 타깃과 DPO chosen/rejected 쌍을 생성합니다.
- **결정적 설정 관리**: `_v2` 전용 경로와 별도 config 파일을 사용해 기존 데이터와 완전히 분리된 실험 환경을 유지합니다.

## 폴더 구조
```text
hallucination-data-synthesizer/
├── README.md                     # 본 문서 (two-utterance branch 전용)
├── configs/
│   ├── default_config.yaml       # 기존 단일 발화 파이프라인 (legacy)
│   └── two_utterances.yaml       # 신규 두 발화 합성용 설정 (직접 생성)
├── docs/
│   └── two_utterances_pipeline.md  # 선택: 상세 설계/QA 노트
├── assets/
│   ├── noises/                   # 노이즈 원본 (공유)
│   ├── zeroth/                   # 기존 데이터 (필요 시 참조)
│   └── zeroth_v2/                # 화자 ID가 포함된 신규 Zeroth 세트
├── data/
│   ├── noise/                    # 노이즈 카탈로그 & 리샘플 캐시
│   ├── zeroth_v2/                # raw_samples, pair 목록 등 신규 메타
│   ├── labels_v2/                # 정렬/증강/레이블 결과(JSONL)
│   └── augmented_audio_v2/       # 합성 WAV
├── scripts/
│   ├── datasets/
│   │   └── export_zeroth_raw_samples.py
│   ├── noise/
│   │   ├── download_aihub_noises.sh
│   │   ├── build_noise_catalog.py
│   │   └── preprocess_noise_resample.py
│   └── pipeline/
│       ├── run_alignment.sh
│       └── run_synthesis.sh
└── src/
    ├── main.py
    ├── pipeline/
    │   ├── step_01_align.py
    │   ├── step_02_augment.py
    │   └── step_03_build_labels.py
    ├── modules/
    └── utils/
```

> `_v2` 디렉터리는 Git에 커밋되지 않으므로(공통 `.gitignore` 적용) 필요 시 수동으로 생성해 사용하세요.

## 설치
- Python 3.10 이상
- `ffmpeg` (시스템 전역 설치)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 빠른 실행 흐름
1. `configs/two_utterances.yaml`을 생성하고 새 데이터 경로·증강 옵션을 정의합니다.
2. 노이즈 데이터(AI Hub)를 다운로드하고 카탈로그/리샘플 캐시를 빌드합니다.
3. Zeroth_v2 원본과 화자 페어 목록을 준비합니다.
4. Step 01 정렬 → Step 02 두 발화 합성 → Step 03 라벨 생성을 실행합니다.
5. `data/labels_v2/<split>/metadata.jsonl`과 `data/augmented_audio_v2/<split>`을 확인합니다.

## 데이터 준비

### 1. 노이즈 자원
```bash
bash scripts/noise/download_aihub_noises.sh
python scripts/noise/build_noise_catalog.py \
  --root assets/noises \
  --output data/noise/noise_catalog.csv
python scripts/noise/preprocess_noise_resample.py \
  --catalog data/noise/noise_catalog.csv \
  --target-dir data/noise/resampled \
  --output data/noise/noise_catalog_resampled.csv \
  --target-sr 16000 --mono
```
카탈로그와 리샘플 결과는 두 발화 실험에서도 그대로 재사용합니다.

### 2. Zeroth_v2 원본
- `assets/zeroth_v2/<split>/<split>_XXXXX.wav`
- `data/zeroth_v2/raw_samples_<split>.jsonl`

`raw_samples` JSONL에는 최소한 다음 필드가 포함되어야 합니다.
```json
{
  "sample_id": "zeroth_v2_train_000123",
  "speaker_id": "spk_045",
  "audio_path": "train/train_000123.wav",
  "text": "...",
  "language": "ko",
  "split": "train",
  "sr_hz": 16000,
  "channels": 1,
  "dataset": "zeroth_korean_v2",
  "index": 123
}
```
기존 `scripts/datasets/export_zeroth_raw_samples.py`를 확장하거나 별도 스크립트를 작성해 `speaker_id`를 포함하도록 변환하세요.

### 3. 발화 페어 매니페스트
두 발화를 결합하기 위해 `data/zeroth_v2/pairs_<split>.jsonl`을 준비합니다. 권장 스키마는 다음과 같습니다.
```json
{
  "pair_id": "spk_045_000123_000456",
  "speaker_id": "spk_045",
  "split": "train",
  "utterances": [
    {"sample_id": "zeroth_v2_train_000123"},
    {"sample_id": "zeroth_v2_train_000456"}
  ],
  "gap_sec": 2.4,
  "notes": {"strategy": "sequential"}
}
```
페어 구성 시 길이, 발화 순서, 텍스트 품질 등을 기준으로 필터링하여 노이즈 삽입에 적합한 조합만 남기세요.

## 파이프라인 단계

### Step 01 – Alignment (`src/pipeline/step_01_align.py`)
- 입력: `data/zeroth_v2/raw_samples_<split>.jsonl`
- 출력: `data/labels_v2/<split>/raw_alignment.jsonl`
- 역할: 각 발화에 대해 WhisperX CTC 정렬을 실행해 단어/토큰 타임스탬프를 계산합니다.
- 체크포인트: `status` 필드가 `ok`인지 확인하고, 실패 레코드는 추후 페어링에서 제외합니다.

### Step 02 – Two-Utterance Augmentation (`src/pipeline/step_02_augment.py`)
- 입력:
  - Step 01 정렬 결과
  - 발화 페어 목록(`pairs_<split>.jsonl`)
  - 노이즈 카탈로그 & WAV
- 처리:
  - 동일 화자 발화 A/B를 불러와 이어 붙일 경계 위치를 계산합니다.
  - `gap_sec` 또는 설정값을 기준으로 중간 구간을 생성하고 노이즈 샘플을 삽입합니다.
  - crossfade, SNR, LUFS, True Peak를 조정해 자연스러운 전이 구간을 만듭니다.
  - 오프셋 맵을 작성해 발화 B 이후 단어 타임스탬프를 재조정합니다.
- 출력:
  - WAV: `data/augmented_audio_v2/<split>/<pair_id>.wav`
  - 메타: `data/labels_v2/<split>/augmented_meta.jsonl`
    - `pair_id`, `speaker_id`, `source_samples`, `augmentation.events`, `offset_map`, `status`
- 실패 시: `status` 값을 `pair_not_found`, `insufficient_gap`, `noise_unavailable` 등으로 남겨 재처리에 활용합니다.

### Step 03 – Label Build (`src/pipeline/step_03_build_labels.py`)
- 입력: 증강 메타, 베이스라인 STT 모델 추론 결과(또는 모듈 내 추론)
- 처리:
  - 보수적/유도 디코딩을 수행해 chosen/rejected 텍스트를 구성합니다.
  - `<SIL_TRANS>` 또는 `<NOISE_TRANS>` 토큰으로 전이 구간을 명시한 SFT 타깃을 생성합니다.
  - WER, hallucination proxy 등 지표를 계산하고 `status` 및 오류 메시지를 기록합니다.
- 출력: `data/labels_v2/<split>/metadata.jsonl`

## 설정 가이드 (`configs/two_utterances.yaml`)
신규 설정 파일은 다음과 같은 구성을 권장합니다.
```yaml
paths:
  input_audio_dir: "./assets/zeroth_v2"
  pair_manifest: "./data/zeroth_v2/pairs_train.jsonl"
  raw_samples_path: "./data/zeroth_v2/raw_samples_train.jsonl"
  alignment_output_dir: "./data/labels_v2"
  augmented_audio_dir: "./data/augmented_audio_v2"
  noise_dir: "./assets/noises"
  noise_catalog: "./data/noise/noise_catalog.csv"
  noise_resampled_dir: "./data/noise/resampled"

aligner:
  model_name: "large-v3"
  align_model_name: "kresnik/wav2vec2-large-xlsr-korean"
  language: "ko"
  device: "cuda"
  rng_seed: 42

pairing:
  min_utterance_sec: 1.5
  max_utterance_sec: 15.0
  max_length_ratio: 3.0
  allow_cross_split: false

synthesis:
  mode: "two_utterances"
  transition_gap_sec:
    default: 2.5
    jitter: 0.5
  crossfade_sec: 0.08
  target_snr_db: 10.0
  loudness_target_lufs: -23.0
  true_peak_dbfs: -1.0
  rng_seed: 9876
  noise_categories: []

labelling:
  baseline_model_name: "openai/whisper-large-v3"
  transition_token: "<SIL_TRANS>"
```
필요 시 Step 02/03 코드에서 새 필드를 읽어 처리하도록 구현하세요.

## 산출물 구조
- `data/augmented_audio_v2/<split>/<pair_id>.wav`: 16 kHz mono 결합 오디오
- `data/labels_v2/<split>/raw_alignment.jsonl`: 단일 발화 정렬 결과
- `data/labels_v2/<split>/augmented_meta.jsonl`: 증강 이벤트 및 오프셋 맵
- `data/labels_v2/<split>/metadata.jsonl`: DPO/SFT 레이블(transition 토큰 포함)

각 JSON 레코드는 `pair_id`, `status`, `error_msg`, `tool_version`, `rng_seed`를 포함해 재현성을 보장해야 합니다.

## QA 및 검증 팁
- 페어 매니페스트에서 `speaker_id`가 일치하는지 사전 검증하세요.
- `augmented_meta.jsonl`을 `jq`로 필터링해 실패 원인을 점검하고, 무작위 샘플의 오디오/스펙트로그램을 청취해 전이 구간 품질을 확인합니다.
- 라벨 빌드 후 chosen/rejected 텍스트에서 전이 토큰이 올바르게 사용되었는지 검토하세요.

## 개발 가이드라인
- 코드 포맷터는 `black`, 린터는 `ruff`(또는 `flake8`)을 사용합니다.
- 모든 함수에 타입 힌트와 Google 스타일 Docstring을 작성하고, 표준 출력 대신 `logging` 모듈을 사용하세요.
- 설정 값은 YAML로 관리하며, 하드코딩을 피합니다.
- 자동화 테스트(`pytest`)와 노트북을 활용해 파이프라인 I/O 계약을 검증하고, RNG 시드 기반 결정성을 주기적으로 확인합니다.

---

이 README는 `feature/two-utterances-augmentation` 브랜치 전용입니다. 기존 단일 발화 파이프라인과 혼동하지 않도록 새 디렉터리와 설정을 사용해 실험을 진행하세요.
