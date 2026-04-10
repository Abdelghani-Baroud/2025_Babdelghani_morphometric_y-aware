
Global-to-Local Guidance for Cortical Sulcal Representation Learning
###########################################################################

This repository aims to apply the self-supervised deep learning pipeline to learn a representation space for each sulcal region. It integrates global information to guide local learning.


Dependencies
------------
- python >= 3.6
- pytorch >= 1.4.0
- numpy >= 1.16.6
- pandas >= 0.23.3


Set up the work environment
---------------------------
First, the repository can be cloned thanks to:

.. code-block:: shell

    git clone https://github.com/neurospin-projects/2023_jlaval_STSbabies/
    cd 2023_jlaval_STSbabies

Then, install a virtual environment through the following command lines:

.. code-block:: shell

    python3 -m venv venv
    . venv/bin/activate
    pip3 install --upgrade pip
    pip3 install -e .

Note that you might need a `BrainVISA <https://brainvisa.info>`_ environment to run
some of the functions or notebooks.

Preterm analysis requires training on UkBioBank, using SimCLR. A comprehensive description is given in contrastive/README.rst.

.. code-block:: shell

    cd contrastive
    python3 train.py mode=encoder

Once the model is trained, the model performances can be assessed using SVC running:

.. code-block:: shell

    cd contrastive
    python3 evaluation/embeddings_pipeline.py

