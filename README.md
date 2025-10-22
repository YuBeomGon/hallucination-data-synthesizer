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
├── requirements.txt
├── configs/
│   └── default_config.yaml
├── scripts/
│   ├── install.sh
│   └── run_synthesis.sh
├── assets/
│   └── noises/
│       └── cafe_noise.wav
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
        └── logging_config.py
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

aligner:
  model_name: "large-v3"
  language: "ko"
  device: "cuda"

synthesis:
  min_gap_ms: 1000
  insertion_duration_ms: 3000
  insertion_type: "noise"  # "silence" 또는 "noise"
  crossfade_ms: 50

labelling:
  baseline_model_name: "openai/whisper-large-v3"
```

## 사용 방법
1. `configs/default_config.yaml`을 프로젝트 환경에 맞게 수정합니다.
2. 전체 파이프라인을 실행합니다.
   ```bash
   bash scripts/run_synthesis.sh
   ```
3. 결과물은 `output_dir`에 지정한 경로에 생성되며, 증강된 오디오와 `metadata.jsonl` 파일이 포함됩니다.

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

## 라이선스
프로젝트에 적용할 라이선스가 정해지지 않았다면, 사용 목적에 맞는 라이선스를 추가하세요.

## 기여
이슈와 풀 리퀘스트는 언제나 환영합니다. 버그 리포트나 기능 제안 시 설정 파일, 사용한 모델 버전, 로그를 함께 제공해 주세요.
