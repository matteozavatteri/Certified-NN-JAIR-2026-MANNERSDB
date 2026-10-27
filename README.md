# Certified Neural Networks: from Verification to Synthesis — MANNERS-DB experiments

This repository contains the code and data for the MANNERS-DB experiments (social appropriateness of robot actions) in:

> Matteo Zavatteri, Davide Bresolin, Nicolò Navarin.
> **Certified Neural Networks: from Verification to Synthesis.**
> *Journal of Artificial Intelligence Research*, 87, 2026.
> DOI: [10.1613/jair.1.22825](https://doi.org/10.1613/jair.1.22825)

```bibtex
@article{ZavatteriBresolinNavarin2026,
  author  = {Zavatteri, Matteo and Bresolin, Davide and Navarin, Nicol{\`o}},
  title   = {Certified Neural Networks: from Verification to Synthesis},
  journal = {Journal of Artificial Intelligence Research},
  volume  = {87},
  year    = {2026},
  doi     = {10.1613/jair.1.22825},
  url     = {https://jair.org/index.php/jair/article/view/22825}
}
```

The paper was published in the JAIR special track [*Integration of Logical Constraints in Deep Learning*](https://www.jair.org/index.php/jair/SpecialTrack-LogicDL).

This repository includes:

- the preprocessing script for the dataset (`preprocessing.py`; the dataset itself is not included, see [Dataset](#dataset)),
- the synthesis tool (`RobotCertifier.py` and `properties/`),
- the raw results of every run reported in the paper (`experiments/`),
- the scripts that turn those results into the paper's tables and figures, and the figures themselves (`img/`).

The other experiments of the same paper are in [Certified-NN-JAIR-2026-ACASXu](https://github.com/matteozavatteri/Certified-NN-JAIR-2026-ACASXu), [Certified-NN-JAIR-2026-HCAS](https://github.com/matteozavatteri/Certified-NN-JAIR-2026-HCAS) and [Certified-NN-JAIR-2026-EP](https://github.com/matteozavatteri/Certified-NN-JAIR-2026-EP).

## What the code does

`RobotCertifier.py` trains a feed-forward ReLU network that predicts how socially appropriate 8 robot actions are in a given home scene. It then uses a counterexample-guided inductive synthesis (CEGIS) loop to make the network satisfy a given property. Each CEGIS iteration does the following:

1. **Train** a network on the current training set: Adam, batch size 100, MSE loss, and early stopping on the validation loss (`training.py`).
2. **Opportunistic search**: evaluate the network on 1000 random valid inputs (1000 random pairs of inputs for $P_2$ and $P_3$) and collect any that violate the property.
3. **Formal verification**: if step 2 finds nothing, export the network to ONNX and ask [Marabou](https://github.com/NeuralNetworkVerification/Marabou) for a counterexample. For $P_2$ and $P_3$, which compare the network's outputs on two different inputs, two copies of the network are exported side by side.
4. **Repair**: fix the counterexamples (at most `--counterexamples`) with Gurobi, changing the scores as little as possible (least squares) so that the property holds.
   - $P_1$ and $P_4$: one small quadratic program per counterexample. $P_4$ is a disjunction, so its program also has two binary variables.
   - $P_2$ and $P_3$: a single quadratic program over the counterexamples and every training pair whose closest human or closest animal is within 0.6 m. Only the mopping score is changed.
5. **Sampling**: add up to `--sampling` noisy copies of each repaired point, keeping only those that still satisfy the property. For $P_2$ and $P_3$, a copy is also dropped if it would violate the property together with another copy or with a training pair.
6. Add the new pairs to the training set and go back to step 1. By default each iteration trains a new network from scratch (use `--keep` to keep training the same one).

The loop stops when Marabou proves that no counterexample exists, so the last network saved is certified for the property.

With `--soft-acceleration` (`-SA` in result names), training also adds a penalty for property violations on random valid inputs, and early stopping uses the validation loss plus that penalty. This usually reduces the number of CEGIS iterations.

The tolerance parameter `--epsilon` (ε) is how far a point must violate the property to count as a counterexample. Unlike the other repositories of the paper, there is no Δ margin here: repair makes the property hold exactly.

## Dataset

The networks are trained on MANNERS-DB, from:

> Jonas Tjomsland, Sinan Kalkan, Hatice Gunes.
> *Mind Your Manners! A Dataset and A Continual Learning Approach for Assessing Social Appropriateness of Robot Actions.*
> Frontiers in Robotics and AI, 9:1–18, 2022.
> [Link to the paper](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.669420/full)

The dataset is not included here. To set it up:

1. Download [`not_normallized.csv`](https://github.com/jonastjoms/MANNERS-DB/blob/master/src/data/not_normallized.csv) into the repository root.
2. Run `python3 preprocessing.py`. This writes `baseline.csv`, which `RobotCertifier.py` reads. It also prints how many rows of the dataset already violate each property.

`preprocessing.py` drops the `Stamp` and `Using circle` columns, turns `Using arrow` into a 0/1 `mode` feature (`mode = 1 − Using arrow`), divides every column by the maximum of its range in `conf.py`, and clips the result to [0, 1]. Values outside the range, such as the large placeholder distances used when there is no child or animal in the scene, become 1.

| Inputs (28) | Range |
|---|---|
| Mode (circle or arrow) | 0/1 |
| Number of people | 0 – 9 |
| Number of people in group | 0 – 5 |
| Group radius | 0.5 – 1 |
| Distance to group | 0 – 6 |
| Robot within group?, Robot facing group? | 0/1 |
| Robot work radius | 0 – 3 |
| Distance to the 1st, 2nd and 3rd closest human | 0 – 6 |
| Direction to the 1st, 2nd and 3rd closest human | 0 – 360 |
| Direction from closest human to robot | 0 – 360 |
| Robot facing the 1st, 2nd and 3rd closest human? | 0/1 |
| 1st, 2nd and 3rd closest human facing robot? | 0/1 |
| Number of children | 0 – 2 |
| Distance to closest child | 0 – 6 |
| Number of animals | 0 – 1 |
| Distance to closest animal | 0 – 6 |
| Number of people sitting/laying in sofa | 0 – 2 |
| Music playing? | 0/1 |
| Number of agents in scene | 1 – 11 |

Distances are in metres and directions in degrees. The 8 outputs are appropriateness scores from 1 to 5 for: vacuum cleaning, mopping the floor, carrying warm food, carrying cold food, carrying drinks, carrying small objects (plates, toys), carrying big objects (tables, chairs), and cleaning (picking up stuff) / starting a conversation.

After a seeded shuffle, the 11,050 samples are split 80/10/10: 8,840 for training, 1,105 for validation and 1,105 for test.

## Properties

Each property is in `properties/P<k>.py`. Each file defines the soft loss, the opportunistic search, the Marabou query, the repair and the sampling. `fm.py` selects the module that matches the property number.

All properties apply to every valid input: all features in [0, 1] and binary features equal to 0 or 1. They are checked on the normalised values, so 0.6 m is 0.1 and 4.5 m is 0.75.

| ID | Property |
|---|---|
| $P_{1}$ | If the closest child is within 0.6 m, mopping the floor has the lowest score of all actions. |
| $P_{2}$ | For any scene where the closest human is within 0.6 m and any scene where the closest animal is within 0.6 m, mopping the floor scores no higher in the first scene than in the second. |
| $P_{3}$ | The same as $P_{2}$ with humans and animals swapped. |
| $P_{4}$ | If the closest child is at least 4.5 m away, carrying warm food or carrying big objects has the highest score. |
| $\mathcal{P}$ = `14` | $P_{1} \land P_{4}$ |

For the combined property, each step (search, verification, repair and sampling) is done separately for $P_1$ and for $P_4$.

## Requirements

- Python 3 with `torch`, `numpy`, `pandas`, `matplotlib` and `onnx`
- [`maraboupy`](https://github.com/NeuralNetworkVerification/Marabou) for formal verification
- [`gurobipy`](https://www.gurobi.com/) with a valid Gurobi licence, for repair
- A LaTeX installation, for `hystogram-counterexamples.py` and `hystogram-sampling.py` (they set `text.usetex = True`)

For example:

```bash
pip install torch numpy pandas matplotlib onnx maraboupy gurobipy
```

Every property module imports Marabou and Gurobi, so both must be installed even to run training alone. A CUDA GPU is used automatically if one is available.

Run every script from the repository root, because the scripts use relative paths.

## Running a single experiment

```bash
python3 RobotCertifier.py 1 \
    --epsilon 0.0001 --learning-rate 0.001 \
    --epochs 100 --early-stopping 20 \
    --hidden-layers 10 --neurons-per-hidden-layer 6 \
    --counterexamples 50 --sampling 20 --seed 100 \
    --soft-acceleration
```

This example uses one of the configurations from the paper. Run `python3 RobotCertifier.py -h` for the full list of options:

| Option | Default | Description |
|---|---|---|
| `property` (positional) | — | Property ID: `1`, `2`, `3`, `4` or `14` |
| `--epsilon` | `0.0001` | ε, the counterexample tolerance |
| `--learning-rate` | `0.001` | Adam learning rate |
| `--epochs` | `100` | Maximum epochs per training phase |
| `--early-stopping` | `100` | Patience, in epochs, on the validation loss |
| `--hidden-layers` | `7` | Number of hidden layers |
| `--neurons-per-hidden-layer` | `8` | Width of each hidden layer |
| `--counterexamples` | `1` | Maximum counterexamples repaired per iteration |
| `--sampling` | `0` | Noisy samples generated around each repaired counterexample |
| `--seed` | `1` | Seed for shuffling and splitting the dataset |
| `--soft-acceleration` | off | Add the soft property loss during training |
| `--keep` | off | Keep training the same network across iterations instead of starting fresh |
| `--cores` | `4` | Number of Marabou workers |
| `--force` | off | Overwrite an existing result directory |

### Output

Each run writes to:

```
experiments/<property>-<epsilon>-<lr>-<layers>x<neurons>-<counterexamples>-<sampling>-<seed>-<L|K>[-SA]/
```

`L` means a new network was trained at each iteration and `K` means the same network was kept (`--keep`). `-SA` marks soft acceleration. The directory contains:

- `network-<i>.pt`: the PyTorch `state_dict` of the network after CEGIS iteration `i`. The highest `i` is the certified network.
- `stats.json`: per-iteration lists `TrainingTIME`, `OpportunisticSeachTIME`, `FormalVerificationTIME`, `RepairTIME`, `SamplingTIME`, `LossOnValidation`, `LossOnTest`, `OpportunisticCEX`, `FormalVerificationCEX`, `NewPairs` and `ModifiedExistingPairs` (training pairs changed by the repair of $P_2$ and $P_3$). It also holds `CEGISiterations`, `OriginalInputs` and `TotalTIME` (in seconds). `TotalTIME` is `null` if the run did not finish. For $P_2$ and $P_3$, counterexamples are counted as pairs of inputs.

If the result directory already exists, the run stops unless you pass `--force`.

## Reproducing the paper

### All experiments

The paper reports 450 runs, all with ε = 0.0001, a 10×6 network, 100 epochs and early stopping after 20 epochs, over every combination of:

- property: `1`, `2`, `3`, `4`, `14`
- learning rate: 0.01, 0.001, 0.0001
- (counterexamples, sampling): (25, 20), (50, 10), (50, 20), (50, 30), (75, 20)
- seed: 100, 200, 300
- with and without `--soft-acceleration`

`Run.py` runs this whole grid one experiment after another, each with a 5-hour timeout and 20 Marabou workers. To launch a smaller grid, edit the lists at the top of the file. The grid the analysis scripts expect is defined in `experiments.py`.

> **Note:** `experiments/` already contains the paper's results, and `RobotCertifier.py` stops on runs whose directory exists. To recompute the results, move `experiments/` somewhere else first.

### Tables and figures

These scripts read `experiments/` and write to the current directory or `img/`:

| Script | Output |
|---|---|
| `python3 experiments.py` | Prints how many runs are present, which are missing and which timed out |
| `python3 latex-counterexamples.py` | `table-counterexamples.tex`: per-property averages (± std) for 25, 50 and 75 counterexamples per iteration (sampling 20) |
| `python3 latex-sampling.py` | `table-sampling.tex`: the same for 10, 20 and 30 samples (50 counterexamples) |
| `python3 hystogram-counterexamples.py` | `img/fv-vs-rep-robot-counterexamples.pdf`: average formal verification time compared with repair time, per property |
| `python3 hystogram-sampling.py` | `img/fv-vs-rep-robot-sampling.pdf`: the same for the sampling runs |
| `python3 convergence.py --counterexamples` | `img/convergence-robot-counterexamples-P<k>.pdf`: number of counterexamples found during the CEGIS iterations, for 25, 50 and 75 counterexamples |
| `python3 convergence.py --sampling` | `img/convergence-robot-sampling-P<k>.pdf`: the same for 10, 20 and 30 samples |
| `python3 convergence.py --baseline` | `img/convergence-robot-baseline-P<k>.pdf`: the same for 50 counterexamples and 20 samples only |

The tables report, for each property and setting, the number of timeouts (TO), CEGIS iterations, times, initial and final losses, counterexamples, calls to formal verification that found a counterexample (FV calls) and new pairs. Rows with soft acceleration are shaded. `convergence.py` loads saved networks at 11 evenly spaced points of each run and counts how many of 1000 random inputs violate the property.

## Repository layout

```
RobotCertifier.py         CEGIS loop (entry point for one experiment)
training.py               Training loop
fm.py                     Selects the property module for each step
conf.py                   Feature names, indices and normalisation ranges
preprocessing.py          Builds baseline.csv from not_normallized.csv
properties/P*.py          One module per property (P1–P4 and the combination P14)
experiments.py            Experiment grid and result loader
Run.py                    Runs the experiment grid
latex-counterexamples.py, latex-sampling.py,
hystogram-counterexamples.py, hystogram-sampling.py,
convergence.py            Scripts that build the tables and figures
marabou.py                Small standalone example of verifying a two-input property with Marabou
milp.py                   Gurobi MILP encoding of a network (not used by the experiments)
experiments/              Raw results of the paper (about 106 MB)
img/                      Figures used in the paper
```
