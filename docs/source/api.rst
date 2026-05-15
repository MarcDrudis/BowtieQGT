API Reference
=============

This section provides detailed API documentation for the BowtieQGT library.

.. toctree::
   :maxdepth: 2
   :caption: API Modules:

   api/bowtieqgt
   api/bowtie_circuits

Core Modules
------------

The BowtieQGT library consists of two main modules:

- **bowtieqgt**: Main computation engine for Quantum Geometric Tensors
- **bowtie_circuits**: Utility functions for circuit manipulation and light-cone analysis

Quick Reference
---------------

Main Classes
~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   bowtie_qgt.bowtieqgt.BowtieQGT

Key Functions
~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   bowtie_qgt.bowtie_circuits.parameter_bowtie
   bowtie_qgt.bowtie_circuits.observable_bowtie
   bowtie_qgt.bowtie_circuits.remove_idle_qwires