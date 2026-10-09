Standalone examples
===================

Run these scripts from the repository root. Install the core library with
``python -m pip install .``; add the extras listed below as needed, for example
``python -m pip install '.[sb3,asp]'``.

.. list-table:: Available examples
   :header-rows: 1

   * - Script
     - Demonstrates
     - Extras
   * - `taxi.py <../examples/taxi.py>`__
     - Tabular Q-learning and norm monitoring in Storm Taxi.
     - None
   * - `merchant.py <../examples/merchant.py>`__
     - Tabular Q-learning with action masks and norm monitoring.
     - None
   * - `gardener.py <../examples/gardener.py>`__
     - DQN or PPO with feature observations.
     - ``sb3``
   * - `pacman.py <../examples/pacman.py>`__
     - DQN or PPO with feature observations and a saved model.
     - ``sb3``
   * - `custom_environment.py <../examples/custom_environment.py>`__
     - Adding labels and a norm monitor to FrozenLake.
     - None
   * - `plot_learning_curves.py <../examples/plot_learning_curves.py>`__
     - Plotting bundled learning-curve CSVs without training.
     - ``plots``
   * - `taxi_policy_fixes.py <../examples/taxi_policy_fixes.py>`__
     - Comparing a learned policy with and without Warn fixes.
     - ``asp``
   * - `merchant_policy_fixes.py <../examples/merchant_policy_fixes.py>`__
     - Comparing a masked policy with and without Environment Friendly fixes.
     - ``asp``
   * - `pacman_often.py <../examples/pacman_often.py>`__
     - Teaching a saved DQN with OFTEN.
     - ``sb3,asp``
   * - `pacman_images.py <../examples/pacman_images.py>`__
     - Image-based DQN with Vegan restraining bolts.
     - ``sb3,render``

Run any script with ``python examples/<filename>``, for example:

.. code-block:: console

   python examples/taxi.py
   python examples/gardener.py
   NPC_GYM_ALGORITHM=ppo python examples/pacman.py

Gardener and feature-based Pacman default to DQN; select PPO with
``NPC_GYM_ALGORITHM=ppo``. They save ``gardener-dqn.zip`` and ``pacman-dqn.zip``
(or ``*-ppo.zip``). Use ``NPC_GYM_MODEL_PATH`` for another new ``.zip`` path;
existing files are refused. Scripts print task outcomes and norm measurements.

Training settings use environment variables, not command-line arguments. Common controls
are ``NPC_GYM_TRAINING_STEPS``, ``NPC_GYM_EVALUATION_EPISODES``,
``NPC_GYM_MAX_EPISODE_STEPS`` and ``NPC_GYM_SEED``. Training examples default to full runs;
feature-based Pacman trains for 5 million steps. Its DQN uses discount factor
0.95 and training rewards divided by 100. See each script for its defaults.

Both ``pacman.py`` and ``pacman_images.py`` accept ``NPC_GYM_LAYOUT=small``, ``medium`` or ``large``.
The corresponding default episode limits are 300, 500 and 800 turns; ``NPC_GYM_MAX_EPISODE_STEPS``
overrides them for training and evaluation. For example::

   NPC_GYM_LAYOUT=large python examples/pacman.py

The OFTEN example uses the small map and requires a matching small-map base checkpoint.

Plotting saved results
----------------------

From a checkout with the bundled experiment results and the ``plots`` extra installed::

   python examples/plot_learning_curves.py

This short script plots all eight Merchant baseline seeds, showing task returns
and the Delivery and Environment Friendly violation counts. It saves
``experiments/output/merchant-learning-curve.png``; existing output is refused.
Edit the paths or count selection to plot other CSVs. See
`learning curves <evaluation.rst#learning-curves>`__ for recording your own results
and the meaning of the shaded bands.

Taxi policy fixing
------------------

.. code-block:: console

   python examples/taxi_policy_fixes.py

Compares the same learned Q-table with and without fixes. Optionally set
``NPC_GYM_OUTPUT_DIR`` to a new directory to retain both evaluations as JSON.
See `policy fixes <policy_fixes.rst>`__ and the
`Taxi catalogue <catalogue/taxi.rst#catalogue-taxi-policy-fixes>`__.

Merchant policy fixing
----------------------

.. code-block:: console

   python examples/merchant_policy_fixes.py

Uses the same output-directory option as Taxi. See the
`Merchant catalogue <catalogue/merchant.rst#catalogue-merchant-policy-fixes>`__
for the norms and planning assumptions.

Feature-based Pacman OFTEN
--------------------------

First save a feature-based DQN, then teach it:

.. code-block:: console

   python examples/pacman.py
   python examples/pacman_often.py

OFTEN loads ``pacman-dqn.zip``; set ``NPC_GYM_MODEL_PATH`` in both commands to use
another path. OFTEN inherits the saved DQN's settings, including discount factor
0.95. Both training phases divide rewards by 100; OFTEN uses ``margin=0.5``.
Evaluation reports raw task rewards. The example compares the base policy, OFTEN,
and continued DQN learning.
``NPC_GYM_TEACHING_STEPS`` defaults to five million transitions per adaptation
(2.5 million ordinary and 2.5 million expert for OFTEN). Each policy is evaluated
on the same 1,000 episode seeds. See the `OFTEN guide <often.rst>`__.
The planner uses the authors' local abstraction; see the
`Pacman catalogue <catalogue/pacman.rst#catalogue-pacman-policy-fixes>`__.

Pixels and restraining bolts
----------------------------

.. code-block:: console

   python examples/pacman_images.py

Uses image observations and norm-based reward adjustments. Set
``NPC_GYM_DEVICE=cpu`` or ``cuda`` to choose a device; the default is ``auto``.
See `restraining bolts <restraining_bolts.rst>`__ for the underlying technique.

Full learning validation
------------------------

On Linux, run the four built-in task examples across three training seeds and
save results in a new directory:

.. code-block:: console

   python experiments/validate_learning.py full --output experiments/output/learning-validation

Use ``--help`` for task selection and CPU allocation options.
