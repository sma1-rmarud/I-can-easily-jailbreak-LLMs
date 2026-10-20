# ACHILLES 초기 점검

이 문서는 코드 복원 전의 점검 기록이다. 이후 추가된 실행 경로와 현재 검증 범위는
[재현 가이드](REPRODUCING.md)를 참고한다.

점검일: 2026-10-20 (Asia/Seoul).
기준: 첨부 논문 `9657_I_can_easily_jailbreak_LLM.pdf`, 현재 작업 트리,
사용자가 지정한 로컬 실험 저장소. 논문은 연구 자료로만 사용했다.
GPU 학습·추론이나 유료 judge API는 실행하지 않았다.

## Git 정리

- 최초 커밋의 `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`을 제거했다.
- 원격 main: `0fc7f78` → `271833cee517558b385359fabf1668a80c16604f`.
- 실제 author/committer는 `sma1-rmarud <rmarud202@uos.ac.kr>`로 유지했다.
- 이전·이후 커밋의 파일 트리는 동일하다. 원격의 예상 커밋을 명시한
  `--force-with-lease`로 push하고 `ls-remote`로 결과를 확인했다.
- 복구용 전체 이력 bundle은 `.git/before-attribution-cleanup.bundle`에 보관했다.
- 기존 README 미커밋 수정과 미추적 PDF는 유지했다.
- AI agent를 저자·공동 저자·기여자로 추가하지 않는 규칙을
  저장소 `AGENTS.md`와 상위 `/data1/home/dmsrud/AGENTS.md`에 기록했다.
- 이 점검 문서와 새 규칙 파일은 로컬 변경이다. GitHub contributor 화면 자체의
  표시 상태는 확인하지 않았다.

## 논문과 현재 공개 코드의 대응

| 논문 구성 | 현재 ACHILLES | 추가로 필요한 것 |
|---|---|---|
| Table 1, Figure 1: 응답 시작 패턴·정렬 데이터 분석 | 전용 재현 진입점 없음 | 분석 데이터 revision, 집계 스크립트, 출력 |
| Table 2: ACHILLES 전후 안전성·성능 | 모델별 GRPO 스크립트, verifier, Llama-Guard ASR 평가 있음 | capability 평가 복원, 표의 각 행과 checkpoint 연결 |
| Table 3: SFT 대조군 | ACHILLES 전용 대조군 recipe 없음 | 실제 full/LoRA 설정 및 학습·평가 결과 provenance |
| Table 5: THETIS 방어 및 LV-RLVR | DPO checkpoint에 대한 공격 스크립트 8–9_2 있음 | THETIS 생성, 통제된 SFT/DPO 학습, Gemini 평가 |
| Figures 2–4, 7, Table 17: 표현 분석 | 전용 코드 발견 못함 | UMAP, tuned lens, layer별 확률·Jeffreys divergence |
| Appendix H: THETIS 구축·검증 | 전용 생성 pipeline 없음 | 필터링·치환·표본 추출·품질 검증 및 데이터 revision |

`open-instruct/`에는 일반 SFT/DPO 구현이 있지만, 그것만으로 논문의 THETIS
통제 실험 recipe가 공개됐다고 볼 수 없다.

## 데이터 점검: 우선 확인 필요

JSONL 전체를 파싱했다. 학습/개발 데이터는 user 메시지를 연결하고,
벤치마크는 평가 코드와 같은 `prompt` 또는 `Goal` 필드를 사용했다.
양끝 공백만 제거한 문자열의 정확 일치이며, 의미적 유사도 검사는 아니다.
행 번호와 파일 SHA-256은 [data_audit.json](data_audit.json)에 기록했다.

| 파일 | 행 수 | 고유 프롬프트 |
|---|---:|---:|
| `data/train/train.jsonl` | 5,000 | 4,981 |
| `data/test/test100.jsonl` | 100 | 100 |
| `data/eval/StrongREJECT/train.jsonl` | 313 | 313 |
| `data/eval/JBB-Behaviors/harmful.jsonl` | 100 | 100 |

학습 데이터의 flag는 unsafe 1,500개, safe 3,500개다.
학습셋에는 동일 프롬프트에 해당하는 초과 행 19개가 있다.

| 비교 | 겹치는 고유 프롬프트 수 |
|---|---:|
| train ↔ StrongREJECT | 11 |
| train ↔ JBB | 1 |
| train ↔ test100 | 3 |
| test100 ↔ StrongREJECT | 100 |
| test100 ↔ JBB | 2 |
| StrongREJECT ↔ JBB | 3 |

`test100` 전부가 StrongREJECT에 포함되어 있다. 학습 중 평가, checkpoint 선택,
최종 평가에 각각 어떻게 사용했는지 확인해야 한다. 이 중복만으로 논문 결과의
유효성이나 영향을 단정할 수는 없다.

현재 train은 `rlvr_latest/data/train_/train_v2.jsonl` 및
`rlvr_dmsrud_03/data/train_/train_v2.jsonl`과 바이트 단위로 동일하다.
test100은 두 경로의 `data/test_/test100_fixed.jsonl`과 동일하다.
StrongREJECT도 `rlvr_latest/data/eval_/StrongREJECT/train.jsonl`과 동일하다.
따라서 확인된 중복은 현재 공개본 정리 과정에서 새로 생긴 차이가 아니다.
기존 데이터를 조용히 덮어쓰지 않고 원 실험 provenance를 먼저 확정해야 한다.

## 복원할 수 있는 로컬 자료

아래 경로는 `/data1/home/dmsrud/` 기준이다.

| 위치 | 확인 내용 | 사용 방법 |
|---|---|---|
| `rlvr_dev/` | HEAD가 정리 전 ACHILLES와 같은 `0fc7f78`; train/test/verifier 동일 | 공개본의 원본 비교 |
| `rlvr_latest/eval_cap_noi/` | setup, run_eval, summary, GPQA/HumanEval 패치 5개 | capability 평가 복원의 우선 후보 |
| `rlvr_dev_old_history.git` | HEAD `b310729`; capability 파일과 Gemini judge 보존 | 고정 commit에서 복원 및 출처 기록 |
| `rlvr_latest/scripts/analyze/safety_judge_gemini.py` | 논문 Figure 5와 대응하는 actionable-assistance 평가 prompt | Table 5 judge 복원 후보 |
| `rlvr_latest/eval_cap_noi/scripts/evaluate/asr/evaluate_asr_gemini_hfmodels.sh` | inference → Gemini 평가 연결 | `data/eval_` 경로 등을 현재 구조에 맞춰 정리 |
| `rlvr_latest/eval_cap_noi/scripts/evaluate/inference/` | OLMo HF 모델 응답 생성 | Gemini 평가 입력 생성 후보 |
| `rlvr_/scripts/filter_dolci_sft.py` | Dolci 필터링의 이전 버전, 샘플 수 5,424 | 최종 논문 설정과 다르므로 참고용 |
| `rlvr_dmsrud_03/` | 과거 학습·평가·SFT 데이터 생성 자료 | Table 3 원 실험 식별에 활용 |
| `rlvr-cache/` | HF/컴파일/실험 캐시 | 코드 원본으로 취급하지 않음 |

capability 핵심 파일, 패치 5개, Gemini judge 및 inference 파일은
`rlvr_latest`와 보관 이력 HEAD에서 바이트 단위로 동일함을 확인했다.
반면 `eval_cap_noi/scripts/evaluate/asr/evaluate_asr_olmo3_instruct.py`는
두 출처 사이에 차이가 있으므로 복원 전에 diff 검토가 필요하다.

capability upstream pin:
`d800e04dcb1ce96791d8b2926cf0cc7703d58457`.
발견한 task 매핑은 다음과 같다.

| 논문 benchmark | 발견한 실행 task |
|---|---|
| GPQA | `gpqa_main_cot_zeroshot_robust` |
| MMLU-Pro | `leaderboard_mmlu_pro` |
| ARC-C | `arc_challenge_chat` |
| MATH | `minerva_math500` |
| HumanEval | `humaneval_think_strip` |

setup은 harness commit을 고정하지만 `lm-eval[vllm]`, `math-verify` 설치의
전체 환경을 고정하지 않는다. run_eval에는 명시적 `--seed`가 없으므로
논문 Table 7과 해당 harness 기본값을 확인하고 실행 manifest에 기록해야 한다.
과거 결과 파일과 위 task의 연결은 아직 확정하지 않았다.

## 논문·문서·구현 사이의 확인 사항

1. **THETIS DPO 규모:** Table 13은 THETIS 32K로 표기하지만 본문 및 Table 16은
   전체 259,922개, chosen 교체 15,129개로 설명한다.
   실제 사용 dataset revision과 학습 로그로 해소해야 한다.
2. **DPO 생성 과정 구분:** `rlvr_latest/scripts/inference/regen_dpo_rejected.py`는
   rejected 응답 재생성용이다. 논문 Appendix H의 chosen 교체 생성기로
   그대로 옮기면 다른 실험을 공개하게 된다.
3. **SFT 이름의 의미:** `27.run_notI_olmo3_sft_original.sh` 등은
   SFT checkpoint에 대한 GRPO 공격이다. THETIS SFT 학습 recipe가 아니다.
4. **평가 checkpoint 누락:** 현재 병렬 ASR registry에서 STAR1-7B는 빈 경로로
   skip되며 14B/32B 및 Table 5의 모델들이 등록되지 않았다.
   step별 공개 checkpoint와 `save_freq 0` 학습 결과 사이의 선택 근거도 필요하다.
5. **judge 명시:** Table 5 전체가 Gemini 평가다. 현재 README는 “THETIS rows”로
   적어 base 비교군에도 같은 judge를 쓴다는 점이 불명확하다.
6. **ASR 판정:** `evaluate_asr_instruct.py`에서 확률 판정과 문자열 판정이 다르면
   문자열을 택하므로 최종 결과는 항상 `verdict.startswith("safe")`와 같다.
   `unsafe_threshold`를 바꿔도 최종 판정은 바뀌지 않는다. 비정상 judge 출력도
   별도 오류로 구분되지 않는다.
7. **환경:** 두 requirements 파일은 NumPy 버전이 다르다
   (`1.26.4` / `2.2.6`). Docker CUDA toolkit과 torch wheel CUDA의 일치 여부도
   실제 build에서 확인해야 한다. 아직 설치·실행 호환성을 검증하지 않았다.
8. **공개 메타데이터:** 루트 LICENSE와 CITATION.cff가 없다.
   vendored `open-instruct/LICENSE`는 있다. README의 Findings 표기와
   BibTeX의 proceedings 표기도 최종 서지정보에 맞춰 정리할 필요가 있다.

## 다음 보완 순서

1. 데이터 중복, 실제 checkpoint 및 Table 13의 DPO 규모부터 provenance 확정.
2. 고정 이력에서 capability 패치·실행기·결과 집계기 복원.
3. Gemini judge와 응답 생성 경로 복원, Table 2/5 judge 설정을 명확히 분리 기록.
4. THETIS 최종 생성·검증 및 통제 SFT/DPO recipe 확보.
5. 논문 표/그림별 명령, 데이터·모델 revision, 결과 파일을 연결하는 manifest 추가.
6. 작은 실행 검증부터 시작해 논문 결과 재현 여부를 별도로 기록.

실시한 검증: Python AST 구문 검사 7개 파일, Bash 구문 검사 21개 파일,
JSONL 파싱 및 정확 일치 중복 검사, 복원 후보 파일 비교, Git tree 동일성,
backup bundle 유효성, 원격 main 확인. 학습 성능·ASR 재현 성공을 뜻하지 않는다.
