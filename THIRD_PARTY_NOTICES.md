# Third-party notices

NPC Gym uses the external sources below. These notices describe that material;
they do not grant a license for NPC Gym as a whole.

## Pacman

The simulation, feature implementation, bundled `small`, `medium` and `large`
maps, and geometric renderer are NPC Gym implementations. The paper experiments
use UC Berkeley's `smallClassic` level in `experiments/layouts/smallClassic.lay`.
The Pacman panel in `envs.png` is rendered by NPC Gym on that level, rather than
by the Berkeley engine.

The level comes from the [UC Berkeley Pacman projects](http://ai.berkeley.edu).
The accompanying source notice is retained here:

> Licensing Information: You are free to use or extend these projects for educational purposes provided that
> (1) you do not distribute or publish solutions, (2) you retain this notice, and (3) you provide clear attribution
> to UC Berkeley, including a link to http://ai.berkeley.edu.
>
> Attribution Information: The Pacman AI projects were developed at UC Berkeley. The core projects and autograders
> were primarily created by John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
> Student side autograding was added by Brad Miller, Nick Hay, and Pieter Abbeel (pabbeel@cs.berkeley.edu).

Ghost schedule proportions follow [The Pac-Man Dossier](https://pacman.holenet.info/),
with repeating cycles scaled to the bundled maps.

## Storm Taxi

Storm Taxi composes the installed [Gymnasium Taxi environment](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/toy_text/taxi.py),
which implements Dietterich's [MAXQ Taxi task](https://doi.org/10.1613/jair.639).
`storm_taxi.py` and `labels.py` retain adaptations of Gymnasium behavior and map
data. Gymnasium is MIT-licensed; its notices are included below.

## Earlier case studies

The normative Pacman and Merchant case studies have external precursors in
[`normative-supervisor`](https://github.com/lexeree/normative-supervisor/tree/0b5061f0ba3a6e0c1876219b310c86e81c394f27).
Gardener's earlier case study is provided by
[`emergency-planning`](https://github.com/S3basuchian/emergency-planning/tree/1f56ec03b25b8037fa41d7069bc4b0d2dc0b38c1).
Both repositories use the MIT license. Their copyright notices are included below.
NPC Gym's environments and normative rules are described in the environment guide
and monitor catalogues; scholarly references there describe the original studies.

## OFTEN and policy fixing

The methodology and adapted reference material come from
[OFTEN-DeepRL: On-the-Fly Teaching of Ethical Norms to Deep Reinforcement Learning Agents](https://doi.org/10.3233/FAIA251200)
and its MIT-licensed [reference implementation](https://gitlab.tuwien.ac.at/martin.tappler/OFTEN-DeepRL/-/tree/3546dfd846f98e7f542b6e6107454bd5e2a5539c),
revision `3546dfd846f98e7f542b6e6107454bd5e2a5539c`.

`policy_fixes/pacman.lp` and the Python adapter adapt the reference's local
coordinate model. `tests/policy_fixes/fixtures/often_pacman.lp` is its unchanged
ASP program, retained as an offline test oracle.
The tabular and SB3 OFTEN implementations adapt the two-stream teaching method
and detached-target expert loss. `tests/integrations/fixtures/often_update.json`
contains numerical outputs from the reference update. No upstream SB3 fork is
bundled. These adapted materials retain the MIT notice below.

## MIT notices

The following notices apply to their respective external material:

- Gymnasium: Copyright (c) 2016 OpenAI; Copyright (c) 2022 Farama Foundation.
- `normative-supervisor`: Copyright (c) 2022 Emery N.
- `emergency-planning`: Copyright (c) 2024 Sebastian Adam.
- OFTEN-DeepRL: Copyright (c) 2025 Lopez Miguel, Ignacio David.

```text
MIT License


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Installed dependencies retain their own licenses.
