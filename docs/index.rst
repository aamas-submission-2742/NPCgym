NPC Gym
=======

NPC Gym provides Gymnasium environments and small, environment-neutral tools for normative reinforcement-learning
research. Environments publish transition labels, monitors count events in label traces, and the evaluator reports
task performance and norm counts without reading private environment state.

Start with the installation instructions and first example in `README.md <../README.md>`__.
Then choose a guide for your task:

* **Describe behavior:** `labeling functions <labeling_functions.rst>`__ turn a reset or step into facts.
* **Measure compliance:** `monitors <monitors.rst>`__ detect and count events; the
  `catalogue <monitor_catalogue.rst>`__ lists supplied rules.
* **Influence learning:** `restraining bolts <restraining_bolts.rst>`__ attach rewards and rule memory.
* **Fix policy decisions:** `policy fixes <policy_fixes.rst>`__ select actions using optional ASP planning.
* **Teach from fixes:** `OFTEN teaching <often.rst>`__ fine-tunes tabular and DQN policies using independent demonstrations.
* **Report results:** `evaluation <evaluation.rst>`__ defines episode measurements and saved results.
* **Compare methods:** `experiment baselines <experiment_baselines.rst>`__ presents measured task performance and norm adherence.
* **Run a complete workflow:** `examples <examples.rst>`__ use the library directly;
  `experiments <experiments.rst>`__ run repository-defined research configurations.

For task dynamics and configuration, see `environments <environments.rst>`__.
For optional training adapters, see `integrations <integrations.rst>`__.
The `public API <api.rst>`__ locates imports, and `monitor specifications <monitor_specifications.rst>`__
defines the accepted rule syntax.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Guides

   environments
   labeling_functions
   monitors
   restraining_bolts
   policy_fixes
   often
   evaluation
   integrations
   experiments
   experiment_baselines
   examples

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Reference

   monitor_catalogue
   monitor_specifications
   api

For external sources and applicable notices, see
`THIRD_PARTY_NOTICES.md <../THIRD_PARTY_NOTICES.md>`__.
