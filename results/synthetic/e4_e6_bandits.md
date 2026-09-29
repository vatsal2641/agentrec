# E4–E6 — bandits & feedback on `synthetic` — ALL NUMBERS FROM THE SIMULATOR

truth: synthetic generator latents; 500 warm users; slate K=5 from top-50 ranked candidates; T=20000 rounds; examination probs [1.0, 0.616, 0.463, 0.379, 0.324].

**Seeds:** E4 and E4b use 3 seeds [0, 1, 2]; E4c, E4c', E5 and E6 use 2 seeds [0, 1], so the same configuration can show slightly different numbers across sections.

## E4a toy 10-armed Bernoulli bandit (T=5000, 20 seeds)

| policy | final regret (mean ± sd) |
|---|---|
| eps_greedy_0.1 | 180.3 ± 102.1 |
| ucb1 | 349.9 ± 27.3 |
| thompson | 101.9 ± 26.2 |

## E4 slate policies (position-as-feature updates, 3 seeds)

| policy | cum. regret (mean ± sd) | exp. clicks/slate | exp. clicks/slate (last 25%) | exploration rate |
|---|---|---|---|---|
| random | 9418.3 ± 6.7 | 0.2758 | 0.2786 | 0.980 |
| frozen | 7177.1 ± 21.7 | 0.3945 | 0.3916 | 0.000 |
| greedy | 4672.8 ± 189.1 | 0.5197 | 0.5238 | 0.000 |
| eps_greedy | 5136.3 ± 76.7 | 0.4996 | 0.5120 | 0.256 |
| linucb | 4740.1 ± 66.1 | 0.5163 | 0.5180 | 0.091 |
| lints | 5061.2 ± 83.4 | 0.4998 | 0.5236 | 0.724 |

## E4b position-bias handling (LinUCB, 3 seeds)

| update mode | cum. regret (mean ± sd) | exp. clicks/slate |
|---|---|---|
| naive | 5570.5 ± 136.1 | 0.4748 |
| position_feature | 4740.1 ± 66.1 | 0.5163 |
| oracle_examined | 4496.5 ± 62.1 | 0.5285 |

## E4c LinUCB alpha (2 seeds)

| alpha | cum. regret (mean ± sd) | exploration rate |
|---|---|---|
| 0.1 | 4869.3 ± 177.8 | 0.011 |
| 0.5 | 4653.2 ± 60.4 | 0.049 |
| 1.0 | 4753.0 ± 77.8 | 0.091 |
| 2.0 | 4735.5 ± 19.0 | 0.141 |

## E4c' LinTS posterior scale v (2 seeds)

| v | cum. regret (mean ± sd) | exploration rate |
|---|---|---|
| 0.05 | 4617.6 ± 55.1 | 0.186 |
| 0.1 | 4614.6 ± 9.6 | 0.295 |
| 0.2 | 4617.7 ± 62.4 | 0.472 |
| 0.5 | 5005.4 ± 32.9 | 0.721 |

## E6 noisy feedback (observed click label flipped with prob p; 2 seeds; mean ± sd)

| p | frozen | greedy | linucb (λ=1) | linucb (λ=100, stronger prior) |
|---|---|---|---|---|
| 0.0 | 7166.0 ± 18.4 | 4711.4 ± 221.8 | 4753.0 ± 77.8 | 4370.4 ± 63.1 |
| 0.1 | 7166.0 ± 18.4 | 4893.0 ± 184.5 | 4812.6 ± 34.7 | 4680.1 ± 240.7 |
| 0.2 | 7166.0 ± 18.4 | 5489.6 ± 133.1 | 5583.8 ± 228.8 | 5022.9 ± 155.8 |
| 0.3 | 7166.0 ± 18.4 | 7017.4 ± 1565.6 | 5834.4 ± 345.4 | 5433.2 ± 290.4 |

## E5 new users, 15 sessions each (200 users, history hidden; 2 seeds)

Arms: `no_adaptation` = ranker top-K, state never updated; `user_state_update` = ranker top-K + user-state updates; `user_state_and_policy_update` = **LinUCB slate selection** + user-state + policy updates (two changes vs the previous arm).

| condition | exp. clicks session 1 | session last | mean over sessions |
|---|---|---|---|
| no_adaptation | 0.1740 | 0.1740 | 0.1740 |
| user_state_update | 0.1740 | 0.3414 | 0.3162 |
| user_state_and_policy_update | 0.1303 | 0.3730 | 0.3153 |
