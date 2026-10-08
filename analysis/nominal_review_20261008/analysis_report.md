# SAC·MBPO nominal 분석

자동 집계: 전체 update event의 정확한 분위수; 10k real-transition 및 5k update 구간. 원본은 수정하지 않음.

## PickCube-v1 / mbpo
경로: `/Users/jihyekim/Desktop/DRL/outputs/nominal_tasks/mbpo_pickcube/mbpo_pickcube_full_500k/seed_0/20261007T143410_ba2d6827`
실행 commit: bdee72d33dc34d7b392d591d6d092f60b1a04e41, dirty=True. 당시 미커밋 소스 동일성 미검증.
500k 카운터 및 audit: 업데이트 248016, audit exit 0.
최종 update: `{'actor_loss': 7036.0908203125, 'critic_loss': 3760420352.0, 'policy_q_min_mean': -6808.72216796875, 'alpha': 55.97014617919922, 'alpha_loss': -3.49019193649292, 'entropy_estimate': -4.062358856201172}`
Frozen 평가: `[{'success_once': 0.0, 'mean_return': 1.4436730172901298, 'mean_episode_length': 50.0, 'checkpoint': '/nfs4/jhkim/repos/DRL/outputs/nominal_tasks/mbpo_pickcube/mbpo_pickcube_full_500k/seed_0/20261007T143410_ba2d6827/checkpoints/final.pt', 'checkpoint_sha256': '1788ebbeb8a8ceca98d8bdf45dd0b87d2cf27f78064707cf70e3df2224f1438a'}]`

## PushCube-v1 / mbpo
경로: `/Users/jihyekim/Desktop/DRL/outputs/nominal_tasks/mbpo_pushcube_500k/mbpo_pushcube_full_500k/seed_0/20261007T094728_f9d45348`
실행 commit: d3e599236160882e47961e6cddfe2798df00ea32, dirty=True. 당시 미커밋 소스 동일성 미검증.
500k 카운터 및 audit: 업데이트 248016, audit exit 0.
최종 update: `{'actor_loss': -3.750339984893799, 'critic_loss': 0.0961759090423584, 'policy_q_min_mean': 3.7617359161376953, 'alpha': 0.0027328492142260075, 'alpha_loss': -0.00046731188194826245, 'entropy_estimate': -4.171039581298828}`
Frozen 평가: `[]`

## StackCube-v1 / mbpo
경로: `/Users/jihyekim/Desktop/DRL/outputs/nominal_tasks/mbpo_stackcube/mbpo_stackcube_full_500k/seed_0/20261007T182651_ab48dfc3`
실행 commit: bdee72d33dc34d7b392d591d6d092f60b1a04e41, dirty=True. 당시 미커밋 소스 동일성 미검증.
500k 카운터 및 audit: 업데이트 248016, audit exit 0.
최종 update: `{'actor_loss': -2.600904941558838, 'critic_loss': 0.00982842966914177, 'policy_q_min_mean': 2.6227216720581055, 'alpha': 0.005206262227147818, 'alpha_loss': -0.000992799294181168, 'entropy_estimate': -4.190704345703125}`
Frozen 평가: `[{'success_once': 0.0, 'mean_return': 31.05088914528489, 'mean_episode_length': 50.0, 'checkpoint': '/nfs4/jhkim/repos/DRL/outputs/nominal_tasks/mbpo_stackcube/mbpo_stackcube_full_500k/seed_0/20261007T182651_ab48dfc3/checkpoints/final.pt', 'checkpoint_sha256': 'ae69956bddb51104d05c4003287138ff9590c20bfea00ec3e73688177ca9dc1c'}]`

## PickCube-v1 / sac
경로: `/Users/jihyekim/Desktop/DRL/outputs/nominal_tasks/sac_pickcube/sac_pickcube_full_500k/seed_0/20261007T124123_37b693e5`
실행 commit: bdee72d33dc34d7b392d591d6d092f60b1a04e41, dirty=True. 당시 미커밋 소스 동일성 미검증.
500k 카운터 및 audit: 업데이트 248016, audit exit 0.
최종 update: `{'actor_loss': -3.5632026195526123, 'critic_loss': 0.008246278390288353, 'policy_q_min_mean': 3.6268155574798584, 'alpha': 0.01623925007879734, 'alpha_loss': 0.001340409042313695, 'entropy_estimate': -3.917454242706299}`
Frozen 평가: `[{'success_once': 1.0, 'mean_return': 4.114706379957497, 'mean_episode_length': 10.13, 'checkpoint': '/nfs4/jhkim/repos/DRL/outputs/nominal_tasks/sac_pickcube/sac_pickcube_full_500k/seed_0/20261007T124123_37b693e5/checkpoints/final.pt', 'checkpoint_sha256': '2d59d569b22f0eef04a3602802ba3060bb6c03faaaebcf268ef722ccfbfd59ed'}]`

## StackCube-v1 / sac
경로: `/Users/jihyekim/Desktop/DRL/outputs/nominal_tasks/sac_stackcube/sac_stackcube_full_500k/seed_0/20261007T163640_e814ffc5`
실행 commit: bdee72d33dc34d7b392d591d6d092f60b1a04e41, dirty=True. 당시 미커밋 소스 동일성 미검증.
500k 카운터 및 audit: 업데이트 248016, audit exit 0.
최종 update: `{'actor_loss': -2.9439468383789062, 'critic_loss': 0.008320123888552189, 'policy_q_min_mean': 2.9604763984680176, 'alpha': 0.004039496183395386, 'alpha_loss': -0.0003694485640153289, 'entropy_estimate': -4.091447830200195}`
Frozen 평가: `[{'success_once': 0.24, 'mean_return': 27.373067382127047, 'mean_episode_length': 43.78, 'checkpoint': '/nfs4/jhkim/repos/DRL/outputs/nominal_tasks/sac_stackcube/sac_stackcube_full_500k/seed_0/20261007T163640_e814ffc5/checkpoints/final.pt', 'checkpoint_sha256': 'c2b3ec0a45fc7d9eaae56a54b744e17c1720a9cde3f703e919af526a1e8eb922'}]`

## 해석 제한
동일 task/seed/contract만 비교한다. 성공 즉시 종료하므로 raw return은 성공률과 함께 해석한다.
고정 normalizer의 초기 데이터/replay와 과거 batch는 저장되지 않았다. checkpoint probe는 한 시점의 한 환경이며 분포를 대표하지 않는다.
생성/보존 pool/실제 batch 진단 coverage가 다르다. ledger audit PASS는 학습 안정성이나 synthetic utility를 입증하지 않는다.
단일 training seed의 nominal 결과이며 dynamics shift·일반적 우월성 증거가 아니다.
