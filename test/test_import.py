# SPDX-License-Identifier: Apache-2.0
# (C) Copyright IBM 2025.

"""Test that the package can be imported."""


def test_import_package():
    """Test that project_name can be imported."""
    import bowtie_qgt

    assert bowtie_qgt is not None


def test_import_version():
    """Test that the version attribute is available."""
    import bowtie_qgt

    assert hasattr(bowtie_qgt, "__version__")
    assert isinstance(bowtie_qgt.__version__, str)
    assert len(bowtie_qgt.__version__) > 0
