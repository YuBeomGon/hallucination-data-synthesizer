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
│   ├── zeroth_v2/                # raw_samples 등 신규 메타
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
1. `configs/two_utterances.yaml`을 작성해 경로·증강 옵션을 정의합니다.
2. 노이즈 데이터(AI Hub)를 다운로드하고 카탈로그/리샘플 캐시를 빌드합니다.
3. `scripts/datasets/export_zeroth_v2.py`로 Zeroth_v2 WAV/JSONL을 준비합니다.
4. `python -m src.pipeline.step_01_two_utterances ...`로 결합 오디오와 `paired_meta.jsonl`을 생성합니다.
5. (선택) Step 02 라벨 빌드를 실행하여 `data/labels_v2/<split>/metadata.jsonl`을 생성합니다.

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
- 새 추출 스크립트:
  ```bash
  python scripts/datasets/export_zeroth_v2.py \
    --dataset kresnik/zeroth_korean \
    --audio-dir assets/zeroth_v2 \
    --raw-samples-dir data/zeroth_v2
  ```
  `--limit` 옵션으로 빠른 샘플 추출을 테스트한 후 전체 데이터를 변환하세요.

## 파이프라인 단계

## 파이프라인 단계

### Step 01 – Two-Utterance Synthesis (신규)
- 실행:
  ```bash
  python -m src.pipeline.step_01_two_utterances \
    --config configs/two_utterances.yaml \
    --split train
  ```
- 입력: `data/zeroth_v2/raw_samples_<split>.jsonl`, 노이즈 카탈로그 & WAV
- 처리:
  - 동일 화자 발화를 길이 제한(<40초 등) 안에서 랜덤 샘플링하고, 필요 시 `time_stretch` 비율을 적용합니다.
  - 발화 A → 전이 노이즈 → 발화 B 순서로 결합하고, pydub 기반 crossfade/smoothing, LUFS/True Peak 보정을 수행합니다.
  - 노이즈 소스·슬라이스 위치·전이 시작/종료 시점·결합 후 총 길이·사용한 RNG 시드를 메타데이터로 기록합니다.
- 출력:
  - WAV: `data/augmented_audio_v2/<split>/<pair_id>.wav`
  - 메타: `data/labels_v2/<split>/paired_meta.jsonl`
    - `pair_id`, `speaker_id`, `source_samples`, `transition`, `noise`, `timings`(utterance A, noise, utterance B), `combined_text`, `status`, `error_msg`
- 실패 시: 길이 초과, 노이즈 미존재 등 사유를 기록하고 `status`를 `skip`/`error`로 설정합니다.
- 구성 옵션과 메타 구조는 `docs/two_utterances_pipeline.md`에서 더 자세히 다룹니다.

### Step 02 – Label Build (`src/pipeline/step_03_build_labels.py`, 확장 예정)
- 입력: Step 01 산출물(WAV + paired_meta), 베이스라인 STT 모델 추론 결과(또는 모듈 내 추론)
- 처리:
  - 보수적/유도 디코딩을 수행해 chosen/rejected 텍스트를 구성합니다.
  - Step 01에서 기록한 전이 구간을 활용하여 `<SIL_TRANS>` 등 전용 토큰을 삽입한 SFT 타깃을 생성합니다.
  - WER, 환각 지표 등을 계산하고 메타정보와 함께 저장합니다.
- 출력: `data/labels_v2/<split>/metadata.jsonl`

## 설정 가이드 (`configs/two_utterances.yaml`)
신규 설정 파일은 다음과 같은 구성을 권장합니다.
```yaml
paths:
  input_audio_dir: "./assets/zeroth_v2"
  noise_dir: "./assets/noises"
  augmented_audio_dir: "./data/augmented_audio_v2"
  metadata_dir: "./data/labels_v2"
  raw_samples_template: "./data/zeroth_v2/raw_samples_{split}.jsonl"
  noise_catalog: "./data/noise/noise_catalog.csv"
  noise_resampled_dir: "./data/noise/resampled"

selection:
  min_utterance_sec: 1.5
  max_utterance_sec: 15.0
  max_total_duration_sec: 40.0
  max_length_ratio: 3.0
  allow_cross_split: false
  rng_seed: 777

synthesis:
  crossfade_sec: 0.1
  target_snr_db: 10.0
  loudness_target_lufs: -23.0
  true_peak_dbfs: -1.0
  transition:
    min_noise_sec: 1.0
    max_noise_sec: 2.5
    min_pause_sec: 0.3
    max_pause_sec: 1.0
    allow_silence_prob: 0.2
    fade_ms: 20
    context_window_sec: 0.75
  time_stretch:
    enable: false
    min_ratio: 0.95
    max_ratio: 1.05
  noise_categories: []

labelling:
  baseline_model_name: "openai/whisper-large-v3"
  transition_token: "<SIL_TRANS>"
  include_silence_token: true
  max_response_sec: 45.0
```
필요 시 Step 01/02 구현에서 위 필드를 참조해 샘플링·전이·라벨 생성 로직을 제어하세요.

## 산출물 구조
- `data/augmented_audio_v2/<split>/<pair_id>.wav`: 16 kHz mono 결합 오디오
- `data/labels_v2/<split>/paired_meta.jsonl`: 결합에 사용된 발화/노이즈/전이 구간 메타데이터
- `data/labels_v2/<split>/metadata.jsonl`: DPO/SFT 라벨 및 품질 지표

각 JSON 레코드는 `pair_id`, `status`, `error_msg`, `tool_version`, `rng_seed` 등을 포함해 재현성을 보장해야 합니다.

## QA 및 검증 팁
- Step 01에서 기록한 `timings`가 실 오디오 길이와 일치하는지 확인하고, 길이 제한(예: 40초)을 넘는 경우 적절히 `skip` 처리됐는지 검토하세요.
- `paired_meta.jsonl`을 `jq`로 필터링해 전이 구간의 노이즈 길이·SNR 분포를 점검하고, 무작위 샘플을 청취해 crossfade 품질을 확인합니다.
- Step 02 결과에서 `<SIL_TRANS>` 등 전이 토큰이 기대 위치에 들어갔는지, chosen/rejected 텍스트가 음성 내용과 일관하는지 확인하세요.

## 개발 가이드라인
- 코드 포맷터는 `black`, 린터는 `ruff`(또는 `flake8`)을 사용합니다.
- 모든 함수에 타입 힌트와 Google 스타일 Docstring을 작성하고, 표준 출력 대신 `logging` 모듈을 사용하세요.
- 설정 값은 YAML로 관리하며, 하드코딩을 피합니다.
- 자동화 테스트(`pytest`)와 노트북을 활용해 파이프라인 I/O 계약을 검증하고, RNG 시드 기반 결정성을 주기적으로 확인합니다.

---

이 README는 `feature/two-utterances-augmentation` 브랜치 전용입니다. 기존 단일 발화 파이프라인과 혼동하지 않도록 새 디렉터리와 설정을 사용해 실험을 진행하세요.
