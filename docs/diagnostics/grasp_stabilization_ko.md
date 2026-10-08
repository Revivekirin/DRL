# PickCube grasp 최소 수정: Phase A–C 구현, 서버 검증 대기

현재 판정: 원인 후보를 검증할 구현을 준비했다. 새 fitting/환경/runtime/learner 테스트는
로컬에서 실행하지 않았다. Phase C 그림·개선 결론은 아직 없으며 서버 실행 결과로 생성한다.
Phase D 및 500k 학습을 시작하지 않는다. 기존 analysis와 미커밋 tracking audit 변경은 보존했다.

## A. 데이터 경로와 증거 범위

1. `envs/maniskill.py`, `envs/state_codec.py`, `envs/maniskill_tasks.py`:
   실제 structured state와 flatten 일치를 검사하고 `extra.is_grasped` 위치를 layout으로 찾는다.
   PickCube의 해당 성분은 관측의 boolean이며 단순 접촉 여부와 같지 않다.
2. `experiments/interaction.py`: GPU vector 성공 종료를 억제한다. 개별 truncation의
   auto-reset 이전 final observation을 복구한 뒤 real replay에 저장한다.
3. `data/model_data.py`: real replay에서만 bounded snapshot을 선택한다. Gaussian 경로는
   observation delta와 reward를 target으로 사용한다. quaternion은 canonical current에
   다음 quaternion 부호를 맞춘 delta다. relative position, goal은 독립적으로 예측한다.
4. `models/probabilistic_ensemble.py`: 첫 train partition에서만 normalizer를 산출하고 고정한다.
   기존 grasp input/target이 일정하면 std=1e-6이다. best 후보에 fit-start member/optimizer가
   들어가며 현재 holdout에서만 비교한다. normalizer를 refit마다 바꾸지 않는다.
5. `algorithms/mbpo/rollouts.py`: real 시작 관측 + 현재 정책 action → elite 예측 → Gaussian
   continuous sampling / 선택된 binary 표현 sampling → quaternion 복원 → finite 검사 → rolling replay.
   큰 finite reward는 여전히 저장 가능하다. reward clipping을 추가하지 않았다.
6. mixed batch는 기존 819 real +205 synthetic /1024. rolling 50,000, 생성2048/refit2000,
   horizon1, gamma0.8, UTD0.5를 변경하지 않았다. synthetic flags=False는 success-suppressed
   학습 계약이며 rollout cutoff를 terminal로 바꾸지 않는다.
7. `algorithms/sac/learner.py`: `r + gamma*(1-terminated)*(min(targetQ)-alpha*logpi)`.
   truncated는 bootstrap한다. loss, optimizer, alpha, target update를 바꾸지 않았다.

과거 분석에서 고정 grasp 스케일의 NLL 지배와 conditional-mean 극단값을 확인했지만
과거 16k–22k batch/model 입력이 없어 완전한 인과 증명이 아니다. source bdee72d...의
실행 당시 dirty patch도 없으므로 현재 코드와 과거 실행의 동일성을 단정하지 않는다.

공식 근거: ManiSkill v3.0.1의
[Panda is_grasping](https://github.com/mani-skill/ManiSkill/blob/v3.0.1/mani_skill/agents/robots/panda/panda.py)
및 [PickCube 관측/reward](https://github.com/mani-skill/ManiSkill/blob/v3.0.1/mani_skill/envs/tasks/tabletop/pick_cube.py).
contact 진단은 양 finger/cube force norm 중 하나라도0.5N 이상인지 측정한다.
Panda grasp는 양쪽 force와 각도를 함께 사용하므로 contact label과 별개다.
현재 관측의 contact만 저장하고 reset 직후 행은 contact 조건부 분석에서 제외한다.

## B. 수정 범위

`model.binary_features`의 기본값은 `legacy_gaussian_delta`다. 기존 config/checkpoint는 이 경로다.
새 opt-in `bernoulli_next_v1`:

- 입력 grasp 0/1을 그대로 사용한다(mean0/std1). 실수로 continuous 입력을 받으면 실패한다.
- target은 delta가 아닌 **다음 grasp의 절대 0/1**이다.
- 기존 mean head의 해당 좌표를 logit으로 사용하고 Bernoulli BCE-with-logits로 학습한다.
  해당 variance 출력은 사용하지 않는다. network 크기와 초기화는 유지된다.
- 다른 좌표의 Gaussian NLL을 그대로 유지하고 binary 좌표 하나만 Bernoulli NLL로 대체하여
  동일 output dimension으로 평균한다. 숨겨진 class weight/oversampling은 없다.
- predict의 해당 mean/variance는 p와 p(1-p). synthetic에는 Bernoulli draw로0/1을 저장한다. 기존 표준정규 draw의 CDF를 uniform으로
  변환하므로 추가 shared RNG draw를 소비하지 않는다.
  deterministic 복원은 p>=0.5의 mode, 진단 raw 복원은 확률 p를 유지한다.
- Boolean 목표가 달라져 Gaussian 총 NLL과 hybrid 총 NLL의 절대값은 동일 척도가 아니다.
  physical reward/state error와 grasp Brier/accuracy를 별도로 비교한다.
- model checkpoint에는 `boolean_policy=bernoulli_next_v1`와 설정을 저장한다.
  새 모델을 legacy codec으로 요구하면 명확히 거부한다. 구 모델의 자동 변환/학습 재개는 없다.
  learner-only 환경 관측 schema는 같으므로 환경 평가 계약은 그대로다.

Quaternion q/-q 처리·정규화·회전각 진단, invariant/관계좌표 모델, reward, elite 선정 절차,
model epoch/lr/patience, SAC/replay는 변경하지 않았다. 선택 objective가 달라져 elite 또는
조기 종료 optimizer step 수가 달라지는 것은 처치의 결과이며 별도 기록한다.
HalfCheetah/PushCube/StackCube에 새 표현을 자동 적용하지 않으며 boolean 없는 task는 opt-in을 거부한다.

의미적 불변조건의 보장 범위: generated 값 finite, quaternion 복원 유효, grasp 정확한0/1.
Goal/관계좌표·grasp와 실제 접촉의 일관성은 아직 보장하지 않는다. 이를 강제 투영하면 별도
변경 요인이 되므로 drift/consistency 진단으로 남긴다. 전체 물리 상태의 타당성을 검증했다고 주장하지 않는다.

## 새 기록과 재현성

공통 runner의 resolved config 저장 시 source SHA/dirty, 실행 source ZIP, tracked dirty patch,
설치 distribution 버전, Python/platform, config, seed를 `run_manifest.json`에 남긴다.
원본 outputs는 덮어쓰지 않는다. 환경변수/API key를 수집하지 않는다.
모델 normalizer는 각 model checkpoint에 있으며 상세 진단을 켜면 refit별 JSONL도 저장한다.

`model.transition_diagnostics=true`는 계측 opt-in이며 기본값false:

- 매 generation의 reward min/max **동일 transition**에 대한 obs/action/member/mean/variance/
  sampled contribution/raw next/projected next를 NPZ에 저장한다.
- 기존 log_every 경계를 지난 첫 실제 update에서 synthetic slot·generation·생성step,
  real/synthetic mask, 실제 batch, TD target, target Q, entropy 항, 두 TD error를 저장한다.
- 추가 policy/model forward나 replay sample을 호출하지 않는다. 실제 update tensor를 읽는다.
- 원래 update ledger에 real/synthetic별 target/TD error min/max/abs-p99를 같은 counter로 남긴다.
  모든 batch를 저장하는 것은 아니며 sampling coverage를 과장하지 않는다.
- 과거 generation에 새로 생긴 계측 값을 소급 생성하지 않는다.

## C. 서버 실행 순서

기존 DRL Python 환경에서 실행한다. 이 문서의 명령은 설치/업로드/추가 RL을 실행하지 않는다.

```bash
cd /nfs4/jhkim/repos/DRL
python -m pytest -q --run-training \
  tests/test_binary_dynamics.py tests/test_model_geometry.py \
  tests/test_mbpo_inference.py tests/test_sac_update.py \
  tests/test_sac_checkpoint.py tests/test_tracking_audit.py
```

정상: 모든 선택 테스트통과. 새 테스트는 네 grasp 전환, binary sampling/표현,
legacy RNG 순서, model checkpoint, best-start NLL, mixed batch RNG 동일성과
실제 learner update의 진단 on/off 결과 동일성을 검사한다. learner/model update가 포함되므로 서버 전용이다.

```bash
POLICY='outputs/nominal_tasks/sac_pickcube/sac_pickcube_full_500k/seed_0/20261007T124123_37b693e5/checkpoints/final.pt'
python -u scripts/diagnose_dynamics.py collect \
  --config configs/runs/mbpo_pickcube_500k.yaml \
  --policy-checkpoint "$POLICY" \
  --transitions 9600 --initial-transitions 4000 \
  --output-root outputs/grasp_diagnostic_collection
```

9600 real transitions=300 vector calls=192 horizon50 episodes, policy updates0.
처음4000은 uniform random이고, 이후 whole-vector episode별 random/frozen deterministic SAC를
번갈아 사용한다. 성공 종료를 억제하고 final observation을 사용한다. SAC checkpoint는 **수집 정책**일 뿐
MBPO 초기화나 resume가 아니다. learner 전체 상태 불변과 update0을 확인한다.
지정한 checkpoint가 서버에 없으면 실제 동일 계약 checkpoint 경로로 바꾼다.
첫 fit의 grasp 분산0은 강제로 만들지 않고 coverage/statistics로 확인한다.

출력된 collection 경로를 정확히 복사해 아래 변수에 넣는다.

```bash
DATA_DIR='<위 collect 명령이 출력한 실제 경로>'
python -u scripts/diagnose_dynamics.py fit \
  --config configs/runs/mbpo_pickcube_500k.yaml \
  --dataset-dir "$DATA_DIR" --initial-transitions 4000 --rounds 5 \
  --output-root outputs/grasp_diagnostic_fit
```

이 명령은 환경을 만들지 않고 real dataset만 읽는다. 두 모델은 seed0 fresh initialization,
동일 데이터/분할/architecture/model 설정으로 비교한다. Round1은 첫4000 transition prefix에서
normalizer를 산출하고, round2–5는 전체9600의 고정 episode split을 사용한다. 이는 정상화 coverage
확장 진단이며 과거 replay의 정확한 재현이 아니다. Round1과 이후 holdout은 다르므로 NLL을
동일 시험셋 개선으로 비교하지 않는다. 전체9600에 대한 조건부 진단은 매 round 같은 표본이다.
실제 데이터 수를 fitting round마다 증가시키지 않는다.

5회씩, ensemble3, 최대5epochs/refit, batch256. fitting 횟수/epoch 한도는 기존 설정 그대로며
각 branch 최대2055 member optimizer steps(두 branch 합계4110; 실제 partition 크기를 함께 확인), early stopping으로 줄 수 있다.
기존500k config는 모델/환경 설정을 읽기 위한 기준이며 이 CLI는 training budget을 실행하지 않는다.

산출물:

- `coverage.json`, `initial_coverage.json`: train/holdout의0→0,0→1,1→0,1→1 개수.
- 각 branch `metrics.jsonl`, `normalizer_round_*.json`, `model_round_*.pt`, `branch_status.json`.
- mean/sample 및 contact/no-contact/grasp 전환별 physical error, raw quaternion norm/회전 오차,
  normalized input magnitude, reward extreme/noise 기여, grasp 정확도.
- `paired_diagnostics.png`: fitting_round x축. initial-prefix와 full partition 차이를 유의한다.
- `gate.json`: coverage와 fitting 완료 여부. **자동 RL 실행 허가는 아니다.**

실패: regression failure, nonfinite 값, checkpoint 계약 불일치, binary 복원 위반,
train/holdout episode 중복, 모델 branch failure. Coverage0 그룹은 미검증이며 성공으로 처리하지 않는다.
Legacy branch가 실패해도 failure를 기록하고 새 branch를 실행한다. `status.json=complete`는
진단 orchestration 종료이고, 실제 fitting 성공은 branch_status/gate로 판단한다.

전달할 파일: 두 run의 manifest/status, dataset_metadata/frozen_collection_check,
coverage/initial_coverage/gate, 두 branch metrics/normalizer/branch_status, paired_diagnostics.png.
원본 dataset/model artifacts도 보존한다. 이 결과로 예측 개선·synthetic 안정·정책 성능을 분리 판정한다.

## Phase D: 아직 실행하지 않음

A–C의 서버 결과를 검토한 뒤20k–50k fresh matched-update SAC/MBPO를 준비한다.
예를 들어20k이면32환경625호출, warmup4000 threshold batch 포함 UTD0.5에 따라8016updates.
같은 evaluation seeds/종료 계약을 쓴다. 기존 SAC 초기화를 자동 사용하지 않는다.
핵심 nonfinite 또는 매우 큰 finite TD/Q/critic을 중지하는 진단 기준은 결과를 본 뒤 **실행 전** 명시한다.
이번에는 Phase D 명령,500k 명령,exact-resume 명령을 제공하지 않는다.
새 clipping/replay/horizon/learning hyperparameter는 추가하지 않았다.
이 비교는 binary 표현이라는 한 묶음의 처치다. 입력 스케일과 target likelihood 중 어느 것의
단독 효과인지는 두 branch만으로 분리할 수 없다. 필요 시 후속 단계에서 한 성분씩 분리한다.
