Experiment baselines
====================

These tables compare task performance and norm compliance in Merchant, Taxi,
Gardener and Pacman. Each policy is evaluated over 1,000 episodes for each of
eight training seeds (0–7), monitoring every norm shown in its environment's
table. The configurations and results below are the paper's benchmark dataset.

Reading the tables
------------------

Values are means ± sample standard deviations across training seeds. Each seed
contributes its episode mean with equal weight; the spread measures variation
between trained policies, not a confidence interval. Return is the undiscounted
task reward, without training penalties or reward scaling. Norm counts are
violations per episode; lower is better. Outcome rates are percentages, and
win and loss rates need not sum to 100% because episodes can time out.

* **Baselines** learn from task rewards alone. A paired base is the pretrained
  policy used by the corresponding fixes or OFTEN method.
* **Policy fixes** apply a planner at each decision without retraining the base.
* **OFTEN** trains the base with policy-fix demonstrations, then evaluates the
  taught policy without a planner.
* **Restraining bolts** train a fresh learner with norm penalties and the active
  automata's states in its observation.

Fixes and OFTEN share evaluation episode seeds with their paired bases.
Standalone baselines and bolts use separate evaluation seed streams, so compare
fixes and OFTEN with their corresponding bases. Training budgets use k for
thousands and M for millions of steps. OFTEN's teaching budget counts
ordinary and expert transitions together, excluding base training. PPO can
slightly exceed its budget because it completes whole rollouts.

Norm definitions are in the catalogues for `Merchant <catalogue/merchant.rst>`__,
`Taxi <catalogue/taxi.rst>`__, `Gardener <catalogue/gardener.rst>`__ and
`Pacman <catalogue/pacman.rst>`__. The `experiment guide <experiments.rst>`__
provides learner settings and training commands.

Merchant — Environment Friendly and DeliveryPacifist
-----------------------------------------------------

Merchant uses action-masked tabular Q-learning for 5M steps, with discount
0.99, learning rate 0.5 and a 150-step episode limit. Epsilon decreases from 1
to 0.2 over the first 10% of training. The baseline, Environment Friendly fixes
and Environment Friendly bolts omit the clock. DeliveryPacifist bolts observe
the clock, making time remaining before sundown available to the learner.
The fixes optimize only Environment Friendly.

Both bolt configurations use minimized automata. Environment Friendly penalizes
violations by 300. DeliveryPacifist uses penalties of 100 for Danger, 1,000 for
CTD and 10,000 for Delivery. CTD (*contrary-to-duty*) counts fighting on the
input immediately after danger. Rewards are unscaled.

Market unload is the percentage of episodes ending with unloading at the market.
Delivery requires reaching the market before sundown; unloading on that visit
is not required. Reaching the market at sundown is late, and unloading later
does not cancel a missed deadline. Pacifist total is Danger + CTD;
DeliveryPacifist total adds Delivery. These totals overlap with their components.

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - Policy
     - Return
     - Market unload (%)
     - Deaths (%)
     - Environment Friendly
     - Delivery
     - Danger
     - CTD
     - Pacifist total
     - DeliveryPacifist total
   * - **Baseline**
     -
     -
     -
     -
     -
     -
     -
     -
     -
   * - 1. Shared baseline
     - 651.3 ± 263.4
     - 86.97 ± 35.17
     - 0.53 ± 1.48
     - 2.532 ± 1.027
     - 0.981 ± 0.037
     - 0.827 ± 0.335
     - 0.022 ± 0.064
     - 0.849 ± 0.352
     - 1.830 ± 0.339
   * - **Environment Friendly fixes and comparisons**
     -
     -
     -
     -
     -
     -
     -
     -
     -
   * - 2. Environment Friendly fixes
     - 215.3 ± 108.9
     - 73.96 ± 39.19
     - 9.09 ± 16.69
     - 0.000 ± 0.000
     - 1.010 ± 0.185
     - 1.410 ± 0.972
     - 0.376 ± 0.692
     - 1.786 ± 1.614
     - 2.796 ± 1.556
   * - 3. Environment Friendly bolts
     - 306.6 ± 13.4
     - 97.94 ± 4.29
     - 2.06 ± 4.29
     - 0.000 ± 0.000
     - 0.557 ± 0.153
     - 1.385 ± 0.115
     - 0.091 ± 0.197
     - 1.476 ± 0.263
     - 2.033 ± 0.194
   * - **DeliveryPacifist bolts**
     -
     -
     -
     -
     -
     -
     -
     -
     -
   * - 4. DeliveryPacifist bolts
     - 750.0 ± 0.0
     - 100.00 ± 0.00
     - 0.00 ± 0.00
     - 2.874 ± 0.135
     - 0.000 ± 0.000
     - 0.755 ± 0.016
     - 0.000 ± 0.000
     - 0.755 ± 0.016
     - 0.755 ± 0.016

Environment Friendly bolts eliminate measured prohibited extraction violations.
DeliveryPacifist bolts achieve market unloading in every evaluated episode,
with no deaths, missed delivery deadlines or CTD violations. Remaining Danger
violations are consistent with the unavoidable crossing's 0.75 attack probability.

Taxi — Emergency and Warn
--------------------------

Taxi uses tabular Q-learning with discount 0.99, learning rate 0.2, no action
masks and a 50-step episode limit. Epsilon decreases from 1 to 0.1 over the
first 10% of training. Baseline and fixes training use 5M steps; Warn-only bolts
use 10M and Emergency bolts use 50M.

The standalone baseline and the base for fixes receive compact observations.
The fixes planner tracks rain history and optimizes only Warn. Bolt learners
observe the full environment state, including home, flood risk and shelter,
plus minimized automaton states. Warn-only bolts penalize Warn by 5; Emergency
bolts penalize each of Warn, Stay, seven-step Safety and three-step Safety by 5.
Training rewards are unscaled.

All five policies monitor the four components. Safety total is the sum of the
two safety counts; Emergency total is Warn + Stay + Safety. These totals overlap
with their components and must not be added to them.

Paired base/fixes evaluations start at seed 10000 + training seed and reset
each episode with successive seeds. Standalone and bolt evaluations start at
10001 + training seed and continue the random stream across episodes.

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - Policy
     - Return
     - Success (%)
     - Warn
     - Stay
     - Safety (7 steps)
     - Safety (3 steps)
     - Safety total
     - Emergency total
   * - **Baseline**
     -
     -
     -
     -
     -
     -
     -
     -
   * - 1. Compact baseline
     - 3.25 ± 0.56
     - 99.96 ± 0.11
     - 0.985 ± 0.007
     - 0.547 ± 0.063
     - 0.172 ± 0.009
     - 0.643 ± 0.038
     - 0.815 ± 0.034
     - 2.347 ± 0.074
   * - **Policy fixes and comparisons**
     -
     -
     -
     -
     -
     -
     -
     -
   * - 2. Base for policy fixes
     - 3.38 ± 0.45
     - 99.94 ± 0.18
     - 0.985 ± 0.006
     - 0.478 ± 0.063
     - 0.187 ± 0.003
     - 0.611 ± 0.021
     - 0.798 ± 0.023
     - 2.261 ± 0.066
   * - 3. Policy fixes (Warn only)
     - 2.23 ± 0.42
     - 99.94 ± 0.18
     - 0.000 ± 0.000
     - 0.457 ± 0.068
     - 0.189 ± 0.002
     - 0.667 ± 0.023
     - 0.856 ± 0.024
     - 1.313 ± 0.066
   * - 4. Warn-only bolt (10M)
     - 0.18 ± 0.34
     - 99.81 ± 0.20
     - 0.048 ± 0.007
     - 0.559 ± 0.042
     - 0.178 ± 0.008
     - 0.812 ± 0.027
     - 0.990 ± 0.028
     - 1.598 ± 0.065
   * - **Full Emergency bolts**
     -
     -
     -
     -
     -
     -
     -
     -
   * - 5. Emergency bolts (50M)
     - 0.51 ± 0.24
     - 99.66 ± 0.18
     - 0.013 ± 0.003
     - 0.007 ± 0.006
     - 0.174 ± 0.009
     - 0.747 ± 0.011
     - 0.921 ± 0.013
     - 0.940 ± 0.017

Warn fixes eliminate measured Warn violations while preserving delivery
success. Emergency bolts reduce Warn and Stay violations, but three-step Safety
accounts for most of their remaining violations.

Gardener — all paper norms
--------------------------

Gardener uses a 1,000-step episode limit. Standalone DQN and PPO train for
100k steps with discount 0.99. DQN uses learning rate 5e-4, a 50k replay buffer,
5k warmup steps, batch size 128, target updates every 2k steps, exploration
fraction 0.3 and a 32–32 network. PPO uses SB3 defaults.

The paired DQN base trains for 250k steps with discount 0.95 and 27 features
that include individual frogs; standalone baselines use task features without
individual frogs. OFTEN adds 250k teaching transitions, using batch size 256
and four learning updates per four ordinary plus four expert transitions.
The four fixes/OFTEN groups share this paired base.

DQN and PPO bolts train for 1M steps with SB3 defaults and discount 0.99.
Their observations combine the 27 frog-aware task features and minimized bolt
states. Penalties are Collect One 10,000, Rescue 1,000 per frog, unpermitted
collection 100 per frog and Drain 10 per puddle. Rewards are unscaled.

Success means reaching the score target of 300. Collect One counts episodes
ending without frog collection, including timeouts. Rescue counts failures
per frog; several obligations for the same frog failing on one step count once.
No Collect counts every frog collected, while unpermitted collection counts
only collections without the permission exception. Drain counts harmful drainage.
Fixes and OFTEN optimize the norms named by their table group; bolts target
Collect One, Rescue, permission-aware collection and Drain.

.. list-table:: Gardener policies; eight seeds and 1,000 episodes per seed
   :header-rows: 1
   :stub-columns: 1

   * - Method
     - Return
     - Success (%)
     - Steps/episode
     - Collect One
     - Rescue per frog
     - Unpermitted collection
     - Drain
     - No Collect
   * - **Baselines**
     -
     -
     -
     -
     -
     -
     -
     -
   * - DQN (100k, γ=0.99)
     - 279.64 ± 0.18
     - 100.00 ± 0.00
     - 223.03 ± 1.60
     - 0.282 ± 0.014
     - 0.642 ± 0.024
     - 0.773 ± 0.020
     - 0.947 ± 0.031
     - 1.008 ± 0.021
   * - PPO (100k, γ=0.99)
     - 279.73 ± 0.09
     - 100.00 ± 0.00
     - 222.42 ± 0.71
     - 0.290 ± 0.017
     - 0.668 ± 0.031
     - 0.765 ± 0.036
     - 0.963 ± 0.027
     - 0.996 ± 0.028
   * - DQN for OFTEN (250k, γ=0.95)
     - 279.72 ± 0.08
     - 100.00 ± 0.00
     - 223.80 ± 1.07
     - 0.267 ± 0.010
     - 0.666 ± 0.040
     - 0.776 ± 0.015
     - 0.988 ± 0.044
     - 1.031 ± 0.021
   * - **No Collect**
     -
     -
     -
     -
     -
     -
     -
     -
   * - DQN + fixes
     - 279.11 ± 0.10
     - 100.00 ± 0.00
     - 229.91 ± 0.93
     - 0.995 ± 0.001
     - 2.207 ± 0.107
     - 0.004 ± 0.002
     - 1.953 ± 0.096
     - 0.005 ± 0.001
   * - DQN + OFTEN
     - 278.47 ± 0.17
     - 99.90 ± 0.08
     - 235.34 ± 1.43
     - 0.910 ± 0.023
     - 2.127 ± 0.182
     - 0.063 ± 0.012
     - 1.935 ± 0.156
     - 0.092 ± 0.025
   * - **Drain**
     -
     -
     -
     -
     -
     -
     -
     -
   * - DQN + fixes
     - 278.25 ± 0.11
     - 100.00 ± 0.00
     - 239.30 ± 1.17
     - 0.211 ± 0.006
     - 0.000 ± 0.000
     - 1.010 ± 0.009
     - 0.000 ± 0.000
     - 1.166 ± 0.010
   * - DQN + OFTEN
     - 278.24 ± 0.16
     - 99.99 ± 0.04
     - 239.12 ± 1.38
     - 0.220 ± 0.003
     - 0.004 ± 0.002
     - 0.991 ± 0.009
     - 0.005 ± 0.002
     - 1.148 ± 0.006
   * - **Permission**
     -
     -
     -
     -
     -
     -
     -
     -
   * - DQN + fixes
     - 279.37 ± 0.10
     - 100.00 ± 0.00
     - 227.64 ± 0.96
     - 0.710 ± 0.005
     - 1.692 ± 0.088
     - 0.003 ± 0.001
     - 1.680 ± 0.080
     - 0.334 ± 0.006
   * - DQN + OFTEN
     - 278.95 ± 0.30
     - 99.91 ± 0.10
     - 230.40 ± 1.51
     - 0.633 ± 0.019
     - 1.670 ± 0.150
     - 0.069 ± 0.018
     - 1.668 ± 0.137
     - 0.427 ± 0.028
   * - **Permission + Drain**
     -
     -
     -
     -
     -
     -
     -
     -
   * - DQN + fixes
     - 272.05 ± 0.26
     - 99.42 ± 0.13
     - 295.98 ± 1.64
     - 0.816 ± 0.006
     - 0.042 ± 0.004
     - 0.014 ± 0.003
     - 0.036 ± 0.003
     - 0.202 ± 0.006
   * - DQN + OFTEN
     - 268.02 ± 0.82
     - 98.16 ± 0.36
     - 320.79 ± 4.22
     - 0.789 ± 0.013
     - 0.031 ± 0.013
     - 0.033 ± 0.017
     - 0.028 ± 0.012
     - 0.230 ± 0.017
   * - **Collection and rescue bolts**
     -
     -
     -
     -
     -
     -
     -
     -
   * - DQN + bolts
     - 279.15 ± 0.15
     - 100.00 ± 0.00
     - 228.26 ± 1.48
     - 0.331 ± 0.005
     - 0.979 ± 0.082
     - 0.685 ± 0.041
     - 1.022 ± 0.047
     - 0.907 ± 0.011
   * - PPO + bolts
     - 1.57 ± 40.07
     - 14.42 ± 9.34
     - 937.65 ± 56.69
     - 0.723 ± 0.040
     - 0.126 ± 0.083
     - 0.289 ± 0.090
     - 0.153 ± 0.095
     - 0.348 ± 0.065

DQN bolts reliably complete the task but leave substantial norm violations.
PPO bolts reduce Rescue, unpermitted collection and Drain violations, but most
episodes time out. These budgets do not establish convergence or show that
the penalty weights achieve the intended priority ordering.

Pacman with feature observations
--------------------------------

Pacman uses Berkeley ``smallClassic``, random ghosts and 63 game features.
All reported evaluations use a 300-step episode limit. Baseline DQN and PPO
train for 5M steps with discount 0.99 and SB3 defaults. Vegan and Vegetarian
fixes/OFTEN share the standalone DQN checkpoints. Trapped trains the same
learner with an additional restriction-active observation.

OFTEN uses an adapted budget of 10M teaching transitions and a 500-step teaching
limit; base training uses 300 steps. Bolts train from scratch with SB3 defaults,
discount 0.99 and a 300-step limit, using these budgets:

.. list-table:: Restraining-bolt training steps
   :header-rows: 1
   :stub-columns: 1

   * - Norm base
     - DQN
     - PPO
   * - Vegan
     - 2.5M
     - 5M
   * - Vegetarian Orange
     - 10M
     - 20M
   * - Hungry Vegan
     - 20M
     - 20M
   * - Trapped
     - 20M
     - 20M
   * - VeganConflict
     - 5M
     - 5M
   * - Hungry Vegan Penalty
     - 20M
     - 20M

Bolt observations prepend a separate one-hot state vector for each minimized
automaton to the game features. Trapped bolts also observe the restriction-active
flag. Training rewards, including bolt penalties, are divided by 100; reported
returns use the game reward without penalties. The `bolt recipes
<experiments.rst#pacman-smallclassic-restraining-bolts>`__ specify the punishments.
Eating a ghost earns 200 points, so avoiding it can lower return while improving
win rate.

The six norm columns count blue and orange ghost consumption, missed eating
deadlines (Hungry), forbidden-zone actions (Trapped), missed blue-ghost
obligations and missed pauses after ghost consumption. Norm-base totals can be
computed from their components without counting any component twice.

.. list-table:: Pacman policies; eight seeds and 1,000 episodes per seed
   :header-rows: 1
   :stub-columns: 1

   * - Method
     - Return
     - Wins (%)
     - Losses (%)
     - Food remaining
     - Blue eaten
     - Orange eaten
     - Hungry
     - Trapped
     - Missed blue obligation
     - Missed pause
   * - **Baselines**
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
   * - DQN (γ=0.99)
     - 1450.5 ± 47.2
     - 82.94 ± 2.88
     - 16.92 ± 2.82
     - 4.97 ± 1.15
     - 1.812 ± 0.023
     - 1.816 ± 0.031
     - 0.681 ± 0.050
     - 6.345 ± 4.848
     - 0.222 ± 0.042
     - 3.344 ± 0.093
   * - PPO (γ=0.99)
     - 1477.6 ± 31.8
     - 84.50 ± 3.19
     - 15.50 ± 3.19
     - 3.20 ± 0.55
     - 1.787 ± 0.058
     - 1.790 ± 0.035
     - 0.704 ± 0.061
     - 6.820 ± 4.816
     - 0.269 ± 0.030
     - 3.346 ± 0.087
   * - DQN for fixes/OFTEN (Trapped feature)
     - 1435.1 ± 34.0
     - 82.04 ± 2.05
     - 17.42 ± 2.05
     - 5.22 ± 0.79
     - 1.818 ± 0.028
     - 1.811 ± 0.028
     - 0.609 ± 0.139
     - 4.928 ± 1.020
     - 0.266 ± 0.061
     - 3.318 ± 0.095
   * - **Vegan**
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
   * - DQN + fixes
     - 507.2 ± 162.1
     - 56.83 ± 19.40
     - 22.55 ± 6.77
     - 5.30 ± 1.90
     - 0.113 ± 0.014
     - 0.105 ± 0.010
     - 0.995 ± 0.005
     - 95.862 ± 20.602
     - 2.929 ± 0.145
     - 0.214 ± 0.018
   * - DQN + OFTEN
     - 841.1 ± 86.7
     - 89.04 ± 7.24
     - 7.44 ± 2.93
     - 1.06 ± 0.66
     - 0.020 ± 0.015
     - 0.019 ± 0.013
     - 0.998 ± 0.007
     - 47.979 ± 21.370
     - 0.668 ± 0.064
     - 0.039 ± 0.027
   * - DQN + bolts
     - 890.0 ± 21.6
     - 94.58 ± 1.37
     - 5.39 ± 1.39
     - 0.89 ± 0.34
     - 0.0065 ± 0.0043
     - 0.0066 ± 0.0039
     - 1.000 ± 0.000
     - 36.007 ± 3.635
     - 0.145 ± 0.170
     - 0.011 ± 0.006
   * - PPO + bolts
     - 837.1 ± 21.7
     - 89.16 ± 1.82
     - 10.64 ± 1.79
     - 1.62 ± 0.30
     - 0.0019 ± 0.0014
     - 0.0024 ± 0.0013
     - 1.000 ± 0.000
     - 31.774 ± 1.581
     - 0.0044 ± 0.0043
     - 0.0031 ± 0.0020
   * - **Vegetarian Orange**
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
   * - DQN + fixes
     - 805.7 ± 40.3
     - 65.78 ± 4.11
     - 30.96 ± 2.68
     - 8.68 ± 0.80
     - 1.512 ± 0.044
     - 0.038 ± 0.010
     - 0.838 ± 0.021
     - 23.600 ± 4.107
     - 0.302 ± 0.038
     - 1.518 ± 0.043
   * - DQN + OFTEN
     - 1156.2 ± 128.7
     - 84.99 ± 11.53
     - 12.15 ± 4.46
     - 2.84 ± 1.51
     - 1.856 ± 0.043
     - 0.083 ± 0.027
     - 0.821 ± 0.018
     - 15.722 ± 4.762
     - 0.178 ± 0.054
     - 1.886 ± 0.064
   * - DQN + bolts
     - 1218.1 ± 38.2
     - 89.48 ± 2.35
     - 10.48 ± 2.32
     - 2.02 ± 0.60
     - 1.897 ± 0.040
     - 0.106 ± 0.037
     - 0.854 ± 0.052
     - 13.207 ± 4.818
     - 0.301 ± 0.079
     - 1.913 ± 0.066
   * - PPO + bolts
     - 850.6 ± 27.7
     - 89.98 ± 2.40
     - 10.01 ± 2.39
     - 1.51 ± 0.37
     - 0.0049 ± 0.0023
     - 0.0044 ± 0.0018
     - 1.000 ± 0.000
     - 28.709 ± 1.632
     - 0.0056 ± 0.0060
     - 0.0066 ± 0.0031
   * - **Trapped**
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
   * - DQN + fixes
     - 466.4 ± 79.7
     - 34.45 ± 3.48
     - 59.85 ± 4.68
     - 25.63 ± 1.79
     - 1.036 ± 0.060
     - 1.014 ± 0.065
     - 0.635 ± 0.135
     - 0.00013 ± 0.00035
     - 0.337 ± 0.065
     - 1.754 ± 0.098
   * - DQN + OFTEN
     - -65.0 ± 131.5
     - 0.20 ± 0.28
     - 76.28 ± 6.06
     - 36.86 ± 3.97
     - 0.736 ± 0.192
     - 0.719 ± 0.193
     - 0.425 ± 0.161
     - 0.013 ± 0.027
     - 0.215 ± 0.089
     - 1.090 ± 0.384
   * - DQN + bolts
     - 1015.8 ± 103.5
     - 58.66 ± 7.01
     - 19.95 ± 4.35
     - 11.51 ± 2.25
     - 1.421 ± 0.055
     - 1.414 ± 0.070
     - 0.586 ± 0.093
     - 0.029 ± 0.017
     - 0.127 ± 0.024
     - 2.660 ± 0.118
   * - PPO + bolts
     - 1165.4 ± 82.4
     - 71.50 ± 4.93
     - 21.06 ± 3.38
     - 6.96 ± 1.38
     - 1.405 ± 0.079
     - 1.387 ± 0.083
     - 0.740 ± 0.198
     - 0.100 ± 0.059
     - 0.186 ± 0.079
     - 2.568 ± 0.159
   * - **HungryVegan**
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
   * - DQN + bolts
     - 41.9 ± 75.1
     - 17.06 ± 5.94
     - 23.71 ± 2.58
     - 24.50 ± 3.31
     - 0.044 ± 0.015
     - 0.043 ± 0.013
     - 0.361 ± 0.060
     - 105.216 ± 45.841
     - 0.240 ± 0.061
     - 0.080 ± 0.027
   * - PPO + bolts
     - 884.8 ± 38.3
     - 80.42 ± 3.23
     - 17.22 ± 3.05
     - 3.39 ± 0.86
     - 0.492 ± 0.020
     - 0.512 ± 0.024
     - 0.036 ± 0.015
     - 20.482 ± 26.850
     - 0.330 ± 0.110
     - 0.945 ± 0.035
   * - **VeganConflict**
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
   * - DQN + bolts
     - 872.4 ± 27.0
     - 92.88 ± 1.84
     - 6.98 ± 1.65
     - 1.27 ± 0.52
     - 0.0020 ± 0.0016
     - 0.0012 ± 0.0013
     - 1.000 ± 0.000
     - 33.814 ± 3.866
     - 0.00038 ± 0.00052
     - 0.0026 ± 0.0016
   * - PPO + bolts
     - 842.4 ± 18.2
     - 89.61 ± 1.61
     - 10.35 ± 1.58
     - 1.63 ± 0.28
     - 0.0021 ± 0.0018
     - 0.0019 ± 0.0022
     - 1.000 ± 0.000
     - 32.207 ± 2.056
     - 0.0041 ± 0.0055
     - 0.0032 ± 0.0030
   * - **HungryPenaltyVegan**
     -
     -
     -
     -
     -
     -
     -
     -
     -
     -
   * - DQN + bolts
     - 971.0 ± 58.3
     - 87.28 ± 3.40
     - 9.08 ± 1.49
     - 2.81 ± 0.95
     - 0.444 ± 0.057
     - 0.454 ± 0.069
     - 0.112 ± 0.110
     - 18.497 ± 5.657
     - 0.273 ± 0.044
     - 0.00012 ± 0.00035
   * - PPO + bolts
     - 985.6 ± 60.0
     - 87.42 ± 3.91
     - 12.11 ± 3.74
     - 2.37 ± 1.05
     - 0.526 ± 0.024
     - 0.527 ± 0.023
     - 0.038 ± 0.015
     - 27.182 ± 18.434
     - 0.626 ± 0.295
     - 0.0076 ± 0.0059

Trapped fixes almost eliminate forbidden-zone actions. Trapped OFTEN also
reduces them, but rarely completes the game.

Data and reproduction
---------------------

Results live in
``experiments/results/<environment>/<experiment-id>/seed-<seed>/``. Each run
records its configuration, seeds, checkpoint hashes, timings, learning curves
and per-episode CSV/JSON evaluations. Each environment's ``archive.json`` records
file hashes; checkpoints and process logs stay in ignored working directories.

Use ``train --paper`` to run these configurations, ``evaluate`` for named
checkpoint evaluations and ``retain`` to export their numerical results. The
`experiment guide <experiments.rst#paper-results>`__ gives the complete workflow.
Numerical reproduction requires the recorded software versions; timings and
provenance describe each execution.

.. _retained-data-and-reproduction:

From the repository root, regenerate reports without checkpoints:

.. code-block:: console

   python experiments/run.py report --input experiments/results/taxi \
     --outdir experiments/output/taxi-report
   python experiments/run.py report --input experiments/results/merchant \
     --outdir experiments/output/merchant-report --evaluation-name paper
   python experiments/run.py report --input experiments/results/gardener \
     --outdir experiments/output/gardener-report --benchmark gardener-paper --evaluation-name paper
   python experiments/run.py report --input experiments/results/pacman \
     --outdir experiments/output/pacman-report --benchmark pacman-paper --evaluation-name paper

Generated CSV tables also include training and teaching times and policy-fix
latency. Timings reflect concurrent CPU runs pinned to individual physical
cores, rather than isolated timing benchmarks. OFTEN curves count teaching
transitions from zero; base curves show pretraining separately.
