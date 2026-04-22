# Soft Robot Simulation and Control in SOFA

This project implements two control strategies for a soft pneumatic robotic finger simulated in SOFA:

* A simulation to pass desireable inputs and collect data
* A learning-based controller using a pretrained Deep Lagrangian Network (DeLaN)



## Overview

The simulation combines:

* Soft body dynamics using SOFA and the SoftRobots plugin
* A learned dynamics model (DeLaN)
* Basic control strategies (feedforward)

The robot is actuated via internal pressure and its state is approximated using bending angles along the finger.


## Files

### `example_input_svd.py`

Learning-based controller:

* Loads a pretrained DeLaN model ( can be trained using SVD or just a simple DeLaN.)
* Computes control input from learned dynamics
* Applies feedforward (and optional feedback) control
* Logs data to `input_test.txt`


### `finger.py`

Baseline controller:

* Applies a desired pressure input
* Logs data to a file

## Data Format

Each logged row:

```text
time, pressure, x0, x1, x2, x3, dx0, dx1, dx2, dx3
```

## Requirements

* SOFA (with SofaPython3)
* SoftRobots plugin
* Python packages: numpy, scipy, jax, jaxlib, haiku


## Running

```bash
runSofa finger.py
```

or

```bash
runSofa example_input_svd.py
```

## Notes

* Units: mm, tonne, s, MPa
* The DeLaN controller requires the pretrained model in `./models/`
* Linear FEM is used for stability

