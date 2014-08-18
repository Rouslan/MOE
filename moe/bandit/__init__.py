# -*- coding: utf-8 -*-
"""Bandit directory containing multi-armed bandit implementation in python.

**Files in this package**

* :mod:`moe.bandit.constant`: some default configuration values for ``optimal_learning`` components
* :mod:`moe.bandit.data_containers`: :class:`~moe.bandit.data_containers.SampleArm`
  and :class:`~moe.bandit.data_containers.HistoricalData` containers for passing data to the ``bandit`` library
* :mod:`moe.bandit.epsilon_first`: :class:`~moe.bandit.epsilon_first.EpsilonFirst`
  object for allocating bandit arms and choosing the winning arm based on epsilon-first policy.
* :mod:`moe.bandit.epsilon_greedy`: :class:`~moe.bandit.epsilon_greedy.EpsilonGreedy`
  object for allocating bandit arms and choosing the winning arm based on epsilon-greedy policy.
* :mod:`moe.bandit.epsilon_interface`: a base :class:`~moe.bandit.epsilon_interface.EpsilonInterface`
  object for all bandit epsilon subtypes.
* :mod:`moe.bandit.ucb1`: :class:`~moe.bandit.ucb1.UCB1`
  object for allocating bandit arms and choosing the winning arm based on UCB1 policy.
* :mod:`moe.bandit.ucb1_tuned`: :class:`~moe.bandit.ucb1_tuned.UCB1Tuned`
  object for allocating bandit arms and choosing the winning arm based on UCB1-tuned policy.
* :mod:`moe.bandit.ucb_interface`: a base :class:`~moe.bandit.ucb_interface.UCBInterface`
  object for all bandit UCB subtypes.
* :mod:`moe.bandit.linkers`: linkers connecting ``bandit`` components.

compute the Bandit Epsilon arm allocation and choosing the arm to pull next.

**Major sub-packages**

**interfaces**
:mod:`moe.bandit.interfaces`

A set of abstract base classes (ABCs) defining an interface for interacting with ``bandit``. These consist of composable
functions and classes to allocate bandit arms and choose arm.

"""
