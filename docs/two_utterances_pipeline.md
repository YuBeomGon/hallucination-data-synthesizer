# Two-Utterance Synthesis Pipeline

이 문서는 동일 화자의 두 발화를 결합하고 전이 구간에 노이즈를 삽입하는 Step 01 파이프라인을 설명합니다. 결과물은 DPO/SFT 라벨링을 위한 `paired_meta.jsonl`과 결합 오디오(`augmented_audio_v2`)입니다.

## 1. 준비 사항
- Zeroth_v2 원본: `scripts/datasets/export_zeroth_v2.py`로 생성한 `assets/zeroth_v2/` 및 `data/zeroth_v2/raw_samples_<split>.jsonl`
- 노이즈 자원: `assets/noises/`, `data/noise/noise_catalog.csv` (필요 시 리샘플 캐시)
- 설정 파일: `configs/two_utterances.yaml`

## 2. 실행 명령
```bash
python -m src.pipeline.step_01_two_utterances \
  --config configs/two_utterances.yaml \
  --split train \
  --limit 1000
```
- `--split`: 처리할 Zeroth_v2 분할(`train`, `test` 등)
- `--limit`: 생성할 최대 페어 수(생략 시 가능한 모든 조합을 순회)
- `--meta-output`, `--audio-output`: 기본 경로를 덮어쓰고 싶을 때 사용

## 3. 처리 절차
1. **발화 선택**  
   - 화자별 발화 목록을 RNG로 셔플한 뒤 두 개씩 묶습니다.  
   - `selection` 섹션에서 길이(`min/max_utterance_sec`), 길이 비율(`max_length_ratio`), 총 길이(`max_total_duration_sec`) 제한을 설정합니다.
2. **노이즈 샘플링**  
   - `transition.min_noise_sec~max_noise_sec` 범위에서 길이를 샘플링하고, 노이즈 카탈로그에서 길이가 충분한 클립을 선택합니다.  
   - `allow_silence_prob` 확률로 노이즈 대신 무음을 삽입할 수 있습니다.  
   - 컨텍스트(발화 A/B 경계 ±`context_window_sec`) RMS/LUFS를 측정해 목표 SNR(`target_snr_db`)에 맞게 스케일합니다.
3. **전이 합성**  
   - 발화 A → 무작위 침묵(`min/max_pause_sec`) → 노이즈 → 발화 B 순으로 배치하고, `crossfade_sec`에 따라 pydub crossfade를 적용합니다.  
   - 노이즈 구간에는 `fade_ms` 길이의 fade-in/out을 추가하고, 전체 결합 오디오는 `loudness_target_lufs`와 `true_peak_dbfs`에 맞춰 정규화합니다.
4. **메타데이터 기록**  
   - `paired_meta.jsonl`에는 다음 정보가 JSONL로 저장됩니다.
     ```json
     {
       "pair_id": "spk_045_a1b2c3_d4e5f6",
       "speaker_id": "spk_045",
       "split": "train",
       "audio": {
         "output_path": "train/spk_045_a1b2c3_d4e5f6.wav",
         "sample_rate": 16000,
         "duration_sec": 33.84,
         "loudness": {
           "lufs_before": -24.1,
           "lufs_after": -23.0,
           "true_peak_dbfs": -1.0,
           "clip_guard_applied": true
         }
       },
       "segments": {
         "utterance_a": {
           "sample_id": "a1b2c3",
           "duration_sec": 12.1,
           "stretch_ratio": 1.02,
           "output_start_sec": 0.0,
           "output_end_sec": 12.1
         },
         "transition": {
           "type": "noise",
           "start_sec": 12.1,
           "end_sec": 15.0,
           "pause_sec": 0.6,
           "crossfade_in_sec": 0.08,
           "crossfade_out_sec": 0.08,
           "noise": {
             "source_path": "cafe/noise.wav",
             "offset_sec": 32.7,
             "duration_sec": 2.3,
             "target_snr_db": 10.0,
             "achieved_snr_db": 9.6
           }
         },
         "utterance_b": {
           "sample_id": "d4e5f6",
           "duration_sec": 18.7,
           "stretch_ratio": 0.99,
           "output_start_sec": 15.0,
           "output_end_sec": 33.8
         }
       },
       "text": {
         "utterance_a": "...",
         "utterance_b": "...",
         "combined": "...",
         "combined_with_token": "... <SIL_TRANS> ..."
       },
       "source_samples": ["a1b2c3", "d4e5f6"],
       "processing": {
         "rng_seed": 777,
         "time_stretch": {"enabled": true, "ratio_a": 1.02, "ratio_b": 0.99}
       },
       "tool_version": {"step_01_two_utterances": "0.1.0"},
       "status": "ok",
       "error_msg": null
     }
     ```

## 4. 실패/스킵 처리
- 총 길이나 길이 비율 제한을 초과하면 `status="skip"`과 함께 `error_msg`에 사유를 적습니다.
- 노이즈를 찾지 못하거나 라이브러리 의존성(librosa 등)이 없으면 `status="error"`로 남기고 다음 페어를 시도합니다.

## 5. 다음 단계
- `paired_meta.jsonl`과 출력 WAV는 Step 02 라벨 생성(`src/pipeline/step_03_build_labels.py` 예정)에서 사용됩니다.
- `<SIL_TRANS>` 삽입 위치는 `segments.transition.start_sec`/`end_sec`를 참조해 결정합니다.
