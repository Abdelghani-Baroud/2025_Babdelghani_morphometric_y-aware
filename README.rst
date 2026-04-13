
Global-to-Local Guidance for Cortical Sulcal Representation Learning
===================================================================

This repository contains the code for learning local cortical sulcal representations
with self-supervised learning while incorporating global morphometric guidance.
The project focuses on training and evaluating representation learning pipelines
for sulcal analysis using PyTorch-based models and contrastive learning tools.

This branch (`MiCCAI_brain_shape`) is dedicated to experiments related to
global brain-shape guidance for local sulcal representation learning.

Repository structure
--------------------

- ``contrastive/``: training, evaluation, and experiment code
- ``AUTHORS.rst``: authorship information
- ``setup.py``: package installation and dependencies

Installation
------------

.. code-block:: bash

   git clone https://github.com/neurospin-projects/2025_Babdelghani_morphometric_y-aware.git
   cd 2025_Babdelghani_morphometric_y-aware
   git checkout MiCCAI_brain_shape

   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -e .

Requirements
------------

Main dependencies are defined in ``setup.py``.
Some scripts may additionally require a BrainVISA environment.

Training
--------

.. code-block:: bash

   cd contrastive
   python3 train.py mode=encoder

Evaluation
----------

.. code-block:: bash

   cd contrastive
   python3 evaluation/embeddings_pipeline.py

Data
----

This project relies on neuroimaging-derived sulcal data and some experiments
require UK Biobank-based pretraining data. Data access is not bundled in this repository.


