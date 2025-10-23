# Troubleshooting

## WhisperX Alignment Drift (CTC Alignment Collapse)

### 현상 요약
- 중간 단어에서 alignment score가 0에 가까워지고, 이후 단어들이 오디오 끝으로 몰려 비정상적으로 짧아짐
- `speech_regions`는 정상 범위를 주는 반면, word timestamps만 비현실적인 값으로 출력

### 주요 원인
1. **정렬 모델 미지정**: WhisperX 기본 설정은 언어별 align 모델을 자동으로 찾지 못하는 경우가 많음. 한국어에서 영어/멀티모델을 쓰면 실패 확률 상승
2. **문자 집합 불일치**: 일부 Wav2Vec2 CTC 모델은 자모 기반, 일부는 완성형 한글 기반. transcript와 align 모델의 vocab이 다르면 강제정렬 실패
3. **세그먼트 구성 미흡**: 긴 텍스트를 한 번에 align하면 중간에서 경로 붕괴 시 뒤 단어가 한꺼번에 끌려감

### 해결 방법
1. **언어 맞춤 align 모델 지정**
   ```yaml
   aligner:
     model_name: "large-v3"              # Whisper tokenizer용
     align_model_name: "kresnik/wav2vec2-large-xlsr-korean"  # 한국어 CTC 모델
   ```
   - Hugging Face의 한국어 CTC 모델 중 corpus와 궁합이 좋은 것을 선택
2. **문자 집합 확인**
   - 모델이 완성형 한글 기반인지 확인
   - 자모 기반 모델이라면 transcript를 자모로 변환하거나 완성형 모델 사용
3. **세그먼트 분할 고려** (필요 시)
   - VAD로 나눈 구간/문장 단위로 segment를 생성하고 각각 align 수행
4. **score 검사 후 재시도** (선택)
   - `word_segments`의 score가 특정 임계값 이하일 경우 정렬 재시도 또는 구간 skip

### 추가 사례
- `align_model_name`을 `kresnik/wav2vec2-large-xlsr-korean`으로 설정하고 다시 정렬을 수행했지만, 샘플 `062fce2e3a36e053`에서 여전히 `최소화하기` 이후 단어들이 10초 대로 몰리는 현상이 확인됨.
- 정렬 모델이 올바르게 로드되었음에도 불구하고 score가 0 또는 매우 낮은 값을 갖는다면, 해당 위치에서 CTC가 덮어쓰기 실패를 일으킨 것으로 판단할 수 있음.
- 이 경우:
  - 다른 한국어 CTC 모델(예: `kresnik/wav2vec2-large-xls-r-300m-korean`)을 시도
  - 긴 문장을 짧은 문장 단위로 나누거나, 자모/완성형 변환을 적용해 문자 집합 불일치를 줄이기
  - 마지막 수단으로 alignment 품질이 낮은 구간은 증강 대상에서 제외하는 fallback 전략을 고려
