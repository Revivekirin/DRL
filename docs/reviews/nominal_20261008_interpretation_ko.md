# 2026-10-08 실측 해석: SAC 대비 nominal MBPO 저성과

이 해석은 `20261007T124123_37b693e5`, `143410_ba2d6827`,
`163640_e814ffc5`, `182651_ab48dfc3`의 네 run과 보조 PushCube 결과에 한정한다.
자동 집계 CSV와 함께 읽는다. 아래 숫자는 원본 JSONL/평가 CSV/checkpoint에서 추출했다.
학습·알고리즘·config를 수정하거나 재실행하지 않았다.

## 1. 결론과 확신 수준

**PickCube는 수치 불안정과 실패 정책, StackCube는 수치적으로 안정적인 미완성 행동으로 구분된다.**
둘을 같은 원인으로 설명하면 안 된다. 네 run은 500,000 real transitions,
15,625 vector calls, 248,016 policy updates로 일치한다. 최종 100개 review seed
10000–10099와 task별 evaluation contract도 일치하며 중복/누락 seed가 없다.

| 최종 checkpoint | 성공률 | return | 길이 |
|---|---:|---:|---:|
| PickCube SAC | 100% | 4.1147 | 10.13 |
| PickCube MBPO | 0% | 1.4437 | 50.00 |
| StackCube SAC | 24% | 27.3731 | 43.78 |
| StackCube MBPO | 0% | 31.0509 | 50.00 |

모두 learner probe 통과, 상태 불변, 평가 update 0이다. 학습은 GPU32에서 성공 종료를
억제하고, 평가는 CPU1에서 성공 즉시 종료한다. 따라서 StackCube의 MBPO return이
더 크다는 사실은 성공률 우위가 아니다. 실패 정책이 50 step 동안 dense reward를
누적할 수 있다. train과 eval return 역시 종료 정책부터 다르다.

**현재 가장 강한 원인 후보는 PickCube의 초기 coverage 부족과 고정된 grasp 스케일이다.**
checkpoint input/target의 `extra.is_grasped` 평균은 0, std는 약1e-6이다.
이후 입력1 또는 delta ±1은 정규화 공간에서 ±1,000,000 규모가 된다.
이 문제는 normalizer를 refit마다 바꿔서 생긴 복원 불일치가 아니다.
후반 NLL은 사실상 grasp target의 이 스케일에 지배된다(아래 정량 근거).
그러나 이것만으로 과거 모든 극단 transition의 생성 원인과 후반 critic 폭증을
완전히 입증한 것은 아니다. 원본 replay/당시 batch가 없다.

## 2. 실행 버전과 자료 범위

현재 로컬 HEAD는 `98867e9e8b274f1c5dec6246976fe8560b766adb`이며 작업 시작 때
working tree는 깨끗했다. 네 run metadata는
`bdee72d33dc34d7b392d591d6d092f60b1a04e41`, **git_dirty=true**이다.
해당 commit 객체는 로컬에 없고 run 내부에 실행 소스/patch도 찾지 못했다.
따라서 정확한 실행 코드 diff를 복원할 수 없다. 현재 소스의 동작은 로그/checkpoint와
부합하지만 동일 코드였다고 단정하지 않는다. 첨부 ZIP의 main ref로 알려진98867e9와
현재 HEAD가 일치한다는 것만으로 dirty 실행 소스가 복원되지는 않는다.

공통 분석 도구는 각 run의 config/resolved config/metadata/tracking summary를 복사하고
전체 파일 목록, 현재 소스 SHA256, git 상태를 별도 분석 디렉터리에 기록한다.
원본 경로는 run_inventory.csv, checkpoint 파일/키는 각 run의 checkpoint_inventory.json에 있다.
checkpoint의 `replay_persisted=false`, `simulator_persisted=false`와 파일 목록을 확인했다.
초기 real dataset, snapshot indices, rollout 입력, elite draw/noise, 정책 batch와 TD target은 없다.

보조 PushCube MBPO는 `20261007T094728_f9d45348`, dirty=true이다. SAC 보조 run은
`20261006T121153_e8582334`이고 전체 update ledger/resolved config 등이 없다.
그 run의 last-update 로그로 전체 update 분위수를 대체하지 않았다. 서로 다른 버전과
평가 seed protocol이므로 네 핵심 run과 동등한 인과 대조군으로 취급하지 않는다.

## 3. 실제 budget과 기록 정상성

- SAC policy sample: real253,968,384 / synthetic0.
- MBPO policy sample: real203,125,104 / synthetic50,843,280.
  한 batch는819 real+205 synthetic, 실제 synthetic 비율20.01953125%이다.
- MBPO 각각249 refits,509,952 generated transitions. 생성량과 재사용 sampling량은 다르다.
- Member optimizer steps: PickCube92,247, StackCube96,663. SAC update와 합산하지 않는다.
- 네 핵심 run은 현재 원본 audit CLI 재실행에서도 episode aggregation PASS,
  MBPO rolling provenance PASS, SAC는 NOT_APPLICABLE이었다.
- PushCube MBPO의 기존 ledger audit는 PASS지만 env_index와 세대 이력은 과거 미기록이므로
  해당 항목은 UNVERIFIED이다. 보조 SAC는 audit에 필요한 ledger가 없어 미검증이다.
- cloud sync는 이 분석으로 확인하지 않았다. 기록 PASS는 학습 안정성 PASS가 아니다.

분위수는 **모든 policy_update event**를 대상으로 계산했다. real transition10k 및
policy update5k 구간에 대해 median,p95,p99,min,max,mean을 저장했다. 같은 step의 여러
update를 모두 포함한다. train.jsonl은 vector call의 마지막 update이므로 주 분석값으로
사용하지 않았다. Episode 통계는 env_index가 있는 종료 episode 원본을 사용한다.

## 4. PickCube: 시간·세대·update 연결

| real step | refit/연결 | 관측 |
|---:|---|---|
| 4,000–14,016 | 초기6회 fit | grasp delta의 unchanged baseline RMSE가 train/holdout 모두0 |
| 16,000 | refit7, elites[1,0] | 세 member 모두 fit-start 복원. train NLL 약8.87e8/6.34e8/2.76e9; holdout은 약−.48/−.64/−.43 |
| 18,016 | refit8, elites[1,0] | 모두 fit-start 복원. holdout NLL 약1.81e7/2.87e5/6.77e7; 새 reward min−326.521, elite mean min−326.615 |
| 18,176 | update7094, event7711 | critic loss121,784.328; 직전 refit8 |
| 20,000 | refit9, elites[2,1] | holdout NLL 약−.71/−.79/−.89로 낮지만 rollout reward min−3,204.966; elite mean min−3,204.909 |
| 34,016 | refit16, elites[2,1] | 새 reward max57.264; elite mean max57.278 |
| 약90k–230k | 일시 회복 | critic 구간 median이약2.34에서0.01 수준으로 감소; 성공률 회복은 아님 |
| 약240k–320k | 재악화 | critic median이0.080→1.87→288→6,165→146,744→수백만으로 상승 |
| 470k–500k | 지속 폭증 | 마지막10k median약1.220e9; 전체 최대1.164e11 |
| 500,000 | update248016 | critic3.760420352e9, policy Q−6,808.722, alpha55.970, entropy추정−4.062 |

16k의 train grasp delta baseline0.01118034는8000개 binary delta 중 한 개의 크기1 변화와
일치한다. holdout delta baseline은0이다. 입력 grasp1의 개수는 별도 기록이 없어 확정할
수 없다. 20k snapshot의 train/holdout delta baseline이 다시0인 것은 고정 시험 데이터의
개선이 아니다. snapshot/partition이 바뀌었다.

전체 critic loss median은 SAC0.00584, MBPO662.69이다. MBPO p99는약4.489e9이다.
기록된 policy 진단의 NaN/Inf는 없지만 매우 큰 유한값으로도 불안정이 명확하다.
Q는 actor가 평가한 batch의 min(Q1,Q2) 평균이며 실제 반환값과의 calibration 오차가 아니다.
Alpha 증가는 후반 악화와 동반하지만 단독 원인으로 판정하지 않는다. entropy 추정치는
후반에도 대체로 target 부근(약−4)에 있으므로 **entropy collapse라고 부르지 않는다.**

## 5. 고정 normalizer와 모델 선택의 정량 근거

모든 저장된 fitted model checkpoint(최초50,016 이후~final)의 mean/std 배열이 동일했다.
4k 당시 checkpoint/dataset은 없으므로 최초 계산 시점은 현재 코드와 로그로 조건부 추적한다.
initial_model의 normalization은 None이며 초깃값 artifact로 초기 데이터 분포를 복원할 수 없다.

- PickCube floor: input grasp(index18), target grasp18와 goal delta26–28.
  나머지 PickCube pose/quaternion 성분은 final normalizer에서1e-3 미만에 해당하지 않는다.
  모든 좌표의 mean/std와 단위 변화 증폭률은 normalizers.csv에 기록했다.
- StackCube는1e-6 floor 성분이 없다. cubeA z input std0.000774,
  cubeB z0.000443, 상대 z0.000891은 작다. 해당 z delta std도0.000312–0.000640이다.
  floor 여부와 물리적으로 작은 분산은 구분해야 한다.
- PushCube는 고정 goal 좌표/goal delta가 floor에 걸린다. 이것만으로 모든 task가
  실패하지는 않는다. 이후 실제로 변하는 grasp와 불변 목표를 같은 위험으로 취급할 수 없다.

PickCube final grasp RMSE는 member별약0.0741이다. normalized target std1e-6과
현재 Gaussian logvar 상한(약0.5000275), 출력차원43을 대입하면 그 **한 좌표의 오차만으로**
전체 normalized NLL 하한이약3.87e7이 된다. 나머지 logvar 항의 가능한 음수 기여−5까지
빼도 관측 holdout NLL의 **99.99996% 이상**이다. normalizer_nll_bound.csv 참조.
이는 전체 NLL/elite 선택이 다른 제어상 중요한 좌표보다 grasp 변화에 지배될 수 있음을
강하게 지지한다. 물리 단위 RMSE가0.074라는 사실을 전체 모델이 유용하다는 근거로 삼을 수 없다.

추가 **새로운 결정론적 CPU 진단**을 수행했다. 저장된50,016 checkpoint의 한 probe와
저장된 action을 고정하고 grasp만0→1로 바꾸었다. member1(elite)의 reward conditional mean은
0.02131→52.74478, normalized reward mean은−0.848→1667.55가 됐다.
동일 진단의 Gaussian reward 표준편차는최대약0.04058이다. 반면 final checkpoint에서는
동일한 방식의 변경이 이런 크기의 reward를 만들지 않았다. 이는 특정 시점의 input-scale
민감도 근거이자 모든 이후 폭증을 같은 입력 하나로 설명할 수 없다는 반대 근거다.

이 반사실 입력은 물리적으로 유효한 grasp 상태임을 보장하지 않는다. **과거 rollout 재현이
아니며** 16k–22k 모델도 저장되지 않았다. 실제 정상/이상 batch의 normalized 값 분포는
없다. checkpoint_inventory에는 저장된 단일 probe들의 normalized max/p95만 별도 표시했다.
후반 probe로 초기 통계를 재계산하거나 교체하지 않았다. 진단은 torch RNG를 격리했고,
샘플링·learner update·optimizer step은 하지 않았다.

## 6. 모델 진단: 물리 성분, mean과 sample

model_diagnostics.csv는 refit/partition/member별 원본 진단을 펼친 테이블이다.
state RMSE는 원래 좌표에서 aligned delta 오차이며 서로 다른 물리 단위가 섞인다.
reward RMSE는 별도이다. NLL은 정규화 공간이고 raw likelihood와 동일 단위가 아니다.
Member 배열 index와 elite rank를 혼동하지 않는다. 매 refit의 holdout은 그 시점 real replay
snapshot에서 나뉜다. 과거 training에 사용된 transition이 이후 holdout에 들어갈 수 있다.

PickCube final holdout:

- qpos RMSE0.156–0.334 / unchanged0.118; qvel4.45–7.36 / unchanged2.94.
- TCP position0.043–0.103 / baseline0.0264.
- Object position0.0181–0.0233 / baseline0.0167.
- TCP→object 상대좌표0.0495–0.1224 / baseline0.0302.
- 정규화 후 TCP angle MAE0.080–0.109rad / unchanged0.0537rad.
- Object angle0.084–0.135rad / unchanged0.0650rad.
- Reward RMSE0.0558–0.0650. 낮은 reward 평균오차는 드문 극단값이나 나쁜 상태 전이를 배제하지 않는다.

StackCube final holdout:

- qpos0.0280–0.0344 / baseline0.0348; qvel0.717–0.749 / baseline0.892.
- TCP position0.0050–0.0060 / baseline0.0114로 일부 개선.
- CubeA 회전각 MAE약0.033rad / unchanged0.0324로 거의 개선 없음.
- CubeB position약0.00136와 baseline0.00137도 거의 동일.
- Reward RMSE0.0844–0.0876. 전체 state 평균만으로 접촉·release 구간의 유용성을 판단할 수 없다.

Quaternion raw norm과 projection 후 norm은 다른 지표다. 20k PickCube raw TCP norm 최대오차는
약2711.56인데 정규화 후 최대 norm 오차는약5.96e-8이다. 단위길이 복구가 올바른 회전
예측을 뜻하지 않는다. canonicalization의 최대절댓값 pivot 경계는 불연속 후보지만
당시 raw quaternion/transition이 없어 직접 원인으로 확인하지 못했다.

PickCube final 생성 grasp의 [0,1] 이탈 비율은60.55%지만 boolean까지의 평균 거리는
0.000551이다. 16k에는47.7% 이탈이어도 평균 거리는약5.43e-8이다. 비율만 보고 모든
이탈이 큰 오류라고 할 수 없다. 정확한 최대 overshoot는 원본 기록이 없어 계산할 수 없다.
Goal drift는 final 생성 RMSE약1.38e-6; 상대 위치 불일치는 TCP→object약0.0807로 별도다.

## 7. A/B/C와 FIFO 전달 경로

| 계층 | 실제 coverage | PickCube reward 관측 |
|---|---|---|
| A 새 생성 | 매 refit2048개 전체의 집계,249회 | 최소−3204.97; 18,016부터 극단값 |
| B retained pool | refit 후 물리 슬롯 처음256개; 전체50k의 대표 표본 아님 | 기록 최소−0.677; 극단값 미관측이 부재 증거는 아님 |
| C 실제 policy batch | log interval을 건넌 첫 update의 synthetic205개,497회 | 54,016에최소−5.194가 실제 사용 batch에서 관측 |

C는248,016 updates 중497개(약0.20%)만 기록한다. 실제 사용된−326/−3205 transition을
특정할 수 없다. A의 elite mean min도 극단적이므로 Gaussian noise만의 문제라는 설명은
지지되지 않는다. 다만 집계 extrema는 동일 transition/member의 paired 수치가 아니므로
두 최소값의 차이를 그 transition의 noise라고 계산해서는 안 된다.

Refit8(18,016 생성)은66,016에서 일부가 남고68,000에서 완전히 퇴출된다.
Refit9(20,000)는68,000 일부 보존,70,016 완전 퇴출.
Refit16(34,016)은82,016 일부 보존,84,000 완전 퇴출이다.
**특정 극단 transition의 슬롯 index는 없으므로 정확한 퇴출 시점은 모른다.**
단위 transition이 세대의 어느 위치에 있었는지에 따라 마지막 두 refit 사이에 사라진다.

이상 세대가 약5만 real transitions 동안 재사용 가능한 구조는 초기 충격의 지속 후보이다.
하지만 250k 이후 새 생성 reward 범위는약−1.974~3.321이고 초기 극단 세대는 이미 퇴출됐다.
그러므로 초기 reward 하나가 replay에 계속 남아서500k 폭증을 설명한다는 주장은 틀리다.
이미 왜곡된 critic/target, 잘못된 next state, 학습 분포 변화에 의한 bootstrap 증폭은 후보지만
과거 batch별 target Q·entropy 항·TD error가 없어 직접 분해할 수 없다.

## 8. 현재 코드 경로 검토

현재 HEAD와 실행 source 동일성 제한을 전제로 다음을 확인했다.

1. `experiments/interaction.py`는 real step의 final observation을 복원하고 개별 flags를 저장한다.
   Truncated는 bootstrap되고 terminated만 차단한다.
2. `data/model_data.py`는 real replay에서 최대10,000개를 without-replacement 추출하고
   앞20%를 holdout으로 나눈다. Synthetic은 모델 fitting에 들어가지 않는다.
3. `models/quaternion.py`에서 input canonicalization, next quaternion sign alignment,
   delta target을 적용한다. 다른 성분·상대좌표·reward는 독립 Gaussian 출력이다.
4. `probabilistic_ensemble.py` normalizer는 **최초 fit train partition에서 한 번만** 계산한다.
   매 refit 변경하지 않는다. Fit-start 후보/최선 epoch snapshot에는 member weights와
   해당 optimizer가 포함된다. normalizer는 고정돼 있어 snapshot별 복원할 새 값이 없다.
   Ensemble RNG/train_steps는 되감지 않으며 전체 checkpoint에는 저장된다.
5. 같은 현재 holdout에서 validation을 비교해 시작 상태 또는 최선 epoch를 복원하고
   현재 member NLL로elite2개를 선택한다. 과거 partition NLL과 직접 비교하지 않는다.
6. Finite NLL이면 fit-start 복원/큰 train loss/나쁜 validation 후에도 refit_count가 증가하고
   synthetic 생성이 진행된다. PickCube는249회 중181회에서 한 member 이상 시작 복원,
   78회는 전부 시작 복원. StackCube는173회/35회다. 복원 자체가 실패 표시나 생성 차단은 아니다.
7. Rollout의 시작 obs는 전체 real replay에서 replacement sampling, action은 현재 stochastic
   actor다. Fit snapshot의 과거 action distribution과도 다르다. 20k의 좋은 holdout과 나쁜
   rollout은 이 coverage 차이와 양립하지만 어느 입력이 원인인지는 저장되지 않았다.
8. normalized mean을 target_std/mean으로 역변환하고 variance에 std²를 곱한다. TS1 elite를
   고른 후 Gaussian noise를 더한다. raw next quaternion은 정규화한다. reward clipping은 없다.
9. finite 검사만 통과하면 매우 큰 유한 next state/reward도 rolling50k replay에 저장된다.
   Horizon1은 계산 cutoff이고 synthetic terminated/truncated는False다. 학습의 성공 종료
   억제와 일치한다. CPU frozen 평가의 terminate-on-success와는 의도적으로 다르다.
10. mixed batch는819real+205synthetic, shuffle 후 SAC에 전달된다. TD target은
    `r + gamma*(1-terminated)*(min(target_Q)-alpha*log_pi)`이고 두 critic MSE 합을 쓴다.
    critic clipping은 없다. 이 식은 확인했지만 당시 transition별 숫자는 복원할 수 없다.

추가 후보: 현재 observation에 robot qvel은 있지만 물체의 linear/angular velocity는 없다.
상대좌표도 독립 예측하므로 pose와 불일치할 수 있다. 접촉·정지 판정의 hidden state가
모델을 어렵게 할 가능성이 있지만, 관측 확장 효과는 이번 로그로 입증할 수 없다.

## 9. 영상: 동일 seed의 관찰과 로그 구분

최종 checkpoint, seed21000/21001, CPU terminate-on-success의 SAC·MBPO 영상에 대해
5개 균등 시점 frame을 추출해 contact sheet를 시각적으로 확인했다. 동영상 전체 frame의
자동 행동 분류를 수행한 것은 아니다. 길이가 다르므로 열은 같은 절대 시간의 대응이 아니다.

- PickCube SAC: 두 seed에서 큐브로 접근하고 큐브가 그리퍼와 함께 상승하는 모습이 보인다.
  영상 episode 로그는각각9/7 step 성공이다.
- PickCube MBPO: 큐브가 테이블에 남고 팔이 목표 작업에서 벗어나는 방향으로 움직이는
  모습이 보인다. 두 영상 로그는50 step 실패다. 이 영상만으로 학습 전체에서 grasp가
  전혀 없었다고 주장하지 않는다.
- StackCube MBPO: 빨간 큐브를 초록 큐브 근처/위로 옮긴 뒤 그리퍼가 그 부근에 머무는
  모습이 보인다. 두 seed 모두50 step 실패다. release 또는 정렬/정지 조건 미충족 후보이나
  영상의 가림과 불연속 frame 때문에 정확한 실패 predicate는 확정하지 않는다.
- StackCube SAC: seed21000은22 step 성공,21001은50 step 실패다. 실패 seed에서도
  적층 위치 부근에 머무는 모습이 있어, MBPO만의 release 문제로 일반화할 수 없다.

학습 seed0의 두 영상은100개 frozen 평가를 대체하지 않는다. 영상 성공 label과 실제
grasp/접촉력 predicate는 구분했다. video_inventory.csv에 원본 경로·episode별 지표를 남겼다.

## 10. 후보별 근거·반대 근거·최소 다음 확인

| 후보 | 지지 근거 | 반대 근거/한계 | 다음 한 요인 확인 |
|---|---|---|---|
| Pick grasp 고정 스케일/coverage | std floor,16k 희귀 delta,final NLL 하한,50k 반사실 mean52.7 | 16–22k 모델/입력 없음; final 반사실은 안정 | 저장 모델의 동일 probe에서 grasp0/1만 바꾸는 진단 재실행; 이후 boolean 입력·target 처리만 별도 설계 |
| Rolling이 초기 충격 재사용 | 세대8/9가 약5만 transition 동안 일부 보존 | 초기 세대는84k 이전 전부 퇴출; 후반 원인 단독 설명 불가 | provenance와 sampled generation ID를 읽기 전용 계측한 뒤 보존 방식 하나만 비교 |
| Bootstrap/상태 불일치 | 초기 next pose norm 폭증, 상대좌표 오차, 후기 Q/critic 폭증 | 과거 batch/TD 분해 없음 | 별도 진단 artifact로 실제 batch·target Q·entropy 항을 저장; reward clipping과 함께 바꾸지 않음 |
| Stack 접촉/release 모델·분포 부족 | object 회전 baseline과 비슷, 영상상 적층 근처 정체 | SAC도 일부 실패; critic 안정, 정보 부족 | 동일 영상 seed에서 success 하위 predicate만 추가 계측; 학습 변경 없음 |
| Normalizer 복원 불일치 | 일반적인 위험 가설 | 현재 코드 normalizer 고정, 저장 배열 동일 | 현재 근거로 우선순위 낮음; 실행 당시 소스 미확보 제한만 유지 |
| Quaternion pivot 경계 | 표현 경계 불연속 가능 | 해당 입력/target 미저장; sign동치 처리 존재 | 저장된 원본 transition 확보 후 q/-q와 경계 근처만 검사 |

이번 분석에서 이미 실행한 최소 진단은 **고정 checkpoint/probe/action에서 boolean 입력 하나만
바꾼 결정론적 실험**이다. 서버에서 같은 공통 분석 CLI로 반복할 수 있다. 아래 명령은 학습,
환경 step, optimizer update, 업로드를 실행하지 않는다.

```bash
cd /nfs4/jhkim/repos/DRL
python tests/test_analysis_tool.py
ANALYSIS_OUT="analysis/nominal_review_$(date +%Y%m%dT%H%M%S)"
python scripts/analyze_runs.py --root outputs/nominal_tasks --aux-root outputs/pushcube_500k \
  --output "$ANALYSIS_OUT" --interpretation docs/reviews/nominal_20261008_interpretation_ko.md
# ffmpeg/ffprobe가 기존 환경에 있는 경우만 영상 frame도 생성:
# 위 명령에 --ffmpeg "$(command -v ffmpeg)" 추가
```

이후 학습 비교를 한다면 한 번에 한 요인만 바꿔야 한다. 우선 boolean feature scale을
입력/target 의미에 맞게 처리하는 후보를 검토하되, reward/critic clipping이나 replay 변경을
동시에 적용하지 않는다. 이는 **아직 구현되지 않은 수정 후보**이며 실행 가능한 옵션처럼
가짜 CLI를 제시하지 않는다. 과거 exact resume도 지원되지 않는다. 현 분석은 추가 fresh
training을 필수로 요구하지 않으며 추가 GPU 검증/학습 명령을 자동 실행하지 않는다.

## 11. 분석 자체의 검증과 남는 제한

분석 전용 unittest로 streaming parsing, 잘못된 JSON line 위치, 정확한 분위수와 부호,
같은 step의 별도 event 보존, generation 경계, streaming audit의 event 누락 검출을 검증했다.
현재 audit CLI는 기존 판정을 유지하면서 update JSON 전체를 메모리에 올리지 않도록
streaming으로 바꿨다. 학습 코드/config는 변경하지 않았다. 각 핵심 run의 audit stdout은
분석 폴더에 보존한다. 이 로컬 분석 실행은 새 학습 runtime 검증을 뜻하지 않는다.

가장 중요한 미확보 자료는 당시 dirty source, 초기 dataset/normalizer 생성 표본,
16k–22k model snapshot, rollout 입력/선택 member/noise, 실제 historical batch/TD terms이다.
보존한 checkpoint의 작은 probe를 실제 replay 분포로 대체하지 않았다. 현재 결론은
단일 training seed의 nominal 비교이며 dynamics-shift synthetic utility 본실험 결론이 아니다.
