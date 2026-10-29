bowtie_qgt.bowtieqgt
====================

.. automodule:: bowtie_qgt.bowtieqgt

Main Class
----------

.. autoclass:: bowtie_qgt.bowtieqgt.BowtieQGT
   :members: evaluate, extract_qgt, extract_gradient, extract_variance
   :member-order: bysource
   :special-members: __init__

Overlap Helpers
---------------

Low-level functions used by :class:`~bowtie_qgt.bowtieqgt.BowtieQGT` to evaluate overlaps
of bowtie states on their shared qubits.

.. autofunction:: bowtie_qgt.bowtieqgt.connected_overlap

.. autofunction:: bowtie_qgt.bowtieqgt.sparse_overlap_tensors

.. autofunction:: bowtie_qgt.bowtieqgt.get_slice

.. autofunction:: bowtie_qgt.bowtieqgt.zeroth
