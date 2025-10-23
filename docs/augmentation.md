# Augmentation Pipeline (Step 02)

이 문서는 Step 02에서 사용되는 노이즈 삽입 기반 증강 알고리즘을 정리합니다.

## 목표
- **자연스러운 삽입**: 클릭/레벨 튐을 최소화하고, 원본 발화의 흐름을 해치지 않습니다.
- **결정성**: 동일한 입력과 RNG 시드를 사용하면 동일한 증강 결과를 얻습니다.
- **경량 경로 우선**: WhisperX 재정렬 없이 offset_map 계산만으로 타임스탬프를 업데이트합니다.

## 입력
- `data/labels/<split>/raw_alignment.jsonl`: Step 01 정렬 결과
- `data/noise/noise_catalog.csv`: 노이즈 메타데이터 (경로, 길이, 카테고리, 샘플레이트 등)
- 설정(`configs/default_config.yaml` → `synthesis` 섹션)
  - `min_gap_sec`: 삽입 가능한 최소 무음 길이 (기본 0.5s)
  - `insertion_duration_sec.min/max`: 삽입 노이즈 길이 범위 (기본 1.5–3.0s)
  - `crossfade_sec`: 좌/우 경계 crossfade 길이 (기본 0.05s)
  - `context_window_sec`: 컨텍스트 RMS/LUFS 추정을 위한 창 길이
  - `target_snr_db`: 말소리 대비 노이즈 SNR 목표 (기본 12dB)
  - `loudness_target_lufs` / `true_peak_dbfs`: 전체 프로그래밍 레벨 관리 값
  - `insertions_per_file`: 파일당 삽입 횟수 (POC는 1)
  - `rng_seed`, `noise_categories`

## 출력
- `data/augmented_audio/<split>/<aug_id>.wav`: 16kHz mono 증강 오디오
- `data/labels/<split>/augmented_meta.jsonl`: 레코드별 메타데이터
  - `aug_id`, `sample_id`, `original_audio_path`, `augmented_audio_path`
  - `augmentation.events[]`
    - `start_sec`/`insert_sec`/`duration_sec`
    - `snr_db`, `crossfade_sec`, `noise_src`, `noise_offset_sec`
  - `augmentation.postprocess`: `lufs_before`, `lufs_after`, `true_peak_dbfs`, `clip_guard_applied`
  - `offset_map`: 삽입으로 인한 시점 이동 정보를 담은 piecewise 매핑
  - `updated_segments`: 삽입 이후 start/end가 이동된 단어 목록(fast path)
  - `status`, `error_msg`

## 알고리즘 요약
1. **후보 gap 탐색**
   - `speech_regions` 사이의 침묵을 우선 사용하고,
   - 부족하면 단어 간 gap 중 `gap ≥ min_gap_sec + 2*crossfade`를 선택합니다 (기본 0.5s).
2. **삽입 길이 결정**
   - `Uniform(min, max)`에서 샘플링하되, gap 안에 crossfade가 들어갈 수 있도록 조정합니다.
3. **노이즈 선택**
   - 카탈로그에서 길이가 충분한 noise clip을 필터링하고, RNG로 선택합니다.
   - clip 내 위치는 `clip_start_sec + offset`으로 랜덤하게 잡습니다.
   - 로딩 시 16kHz mono로 리샘플하고 20ms fade-in/out 적용.
4. **레벨 맞춤**
   - 삽입 지점 전/후 컨텍스트(기본 ±0.75s) RMS/LUFS를 추정합니다.
   - `target_snr_db` 기준으로 노이즈를 스케일링하고, 결과 전체를 `loudness_target_lufs`/`true_peak_dbfs`에 맞춥니다.
5. **Crossfade 삽입**
   - 좌/우 crossfade를 적용하여 이음새를 부드럽게 합니다.
   - 삽입 이후 전체 파형을 clip guard(-1dBFS)로 마무리합니다.
6. **메타데이터 업데이트**
   - `offset_map` 생성: [(0,0), (gap_start,gap_start), (gap_end,gap_end+Δ), (orig_dur, orig_dur+Δ)]
   - `updated_segments`: gap 이후 단어 start/end에 Δ를 더한 목록 (fast path)
   - 실패 시 `status`를 `skip`/`error`로 표기하고 원인을 `error_msg`에 기록합니다.

## 실패/회피 로직
- gap 길이가 부족할 때 `insufficient_gap`
- 카탈로그에서 적절한 노이즈를 찾지 못하면 `skip`
- 노이즈/컨텍스트 RMS가 0에 가까우면 삽입을 생략
- Loudness/True Peak 제한을 초과하면 재스케일링 → 2회 실패 시 `skip`
- 파일당 삽입 횟수 초과 시 나머지는 스킵

## 빠른 경로 vs 정밀 경로
- **빠른 경로(fast path)**: offset_map 기반 이동만 수행, WhisperX 재정렬 없음 (기본)
- **정밀 경로(optional)**: 삽입 이후 WhisperX 재정렬을 추가 실행해 `updated_segments_refined`를 얻을 수 있습니다. 대량 생성에는 권장되지 않으며, 품질 검증 샘플에서만 사용하십시오.

## 검증/QA
- `notebooks/augment_qa.ipynb`를 통해 샘플을 추출하여 before/after 파형, 스펙트로그램, crossfade 구간 확대, LUFS/SNR 지표, 오디오 재생을 확인할 수 있습니다.
- `data/labels/<split>/augmented_meta.jsonl`에서 `status`를 필터링해 실패 비율을 점검하십시오.

## 실행 예시
```bash
conda activate hallucination_synth
python -m src.pipeline.step_02_augment \
  --config configs/default_config.yaml \
  --split train
```

결과 WAV는 `data/augmented_audio/<split>/`, 메타데이터는 `data/labels/<split>/augmented_meta.jsonl`에 생성됩니다.
