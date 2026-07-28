.. _testing:

Testing
=======

Unit tests live in ``tests/`` and run without any services. Functional
tests in ``functional_tests/`` exercise the full gRPC pipeline end-to-end.

Functional Tests
----------------

Functional tests call real clients with sample inputs and verify outputs.
All services (S2S, ASD, LipSync, Controller) must be running before you
execute them.

Test Coverage
~~~~~~~~~~~~~

* **Controller Client** - Orchestrated pipeline testing
* **Direct Client** - Direct service communication testing
* **S2S Client** - Audio translation with latency analysis
* **LipSync Client** - Lip synchronization testing
* **ASD Client** - Active speaker detection testing

Quick Start
~~~~~~~~~~~

.. code-block:: bash

   # Run all functional tests
   python -m pytest functional_tests/ -v

   # Run specific client tests
   python -m pytest functional_tests/test_controller_client.py -v
   python -m pytest functional_tests/test_s2s_client.py -v

Prerequisites
~~~~~~~~~~~~~

Before running functional tests, ensure:

* All services running (S2S, ASD, LipSync, Controller)
* Sample input files in ``assets/``
* Python environment with dependencies

Detailed Testing Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For comprehensive functional testing documentation, test configuration, and troubleshooting, see:

* **Test Structure**: `functional_tests/README.md <../functional_tests/README.md>`_
* **Individual Test Files**: Each client has dedicated test files
* **Output Validation**: Tests validate file format, size, and content
* **Error Handling**: Graceful handling of service unavailability

Test Output Files
~~~~~~~~~~~~~~~~~

All test outputs are preserved in ``functional_tests/outputs/`` directory:

* ``controller_comprehensive_output.mp4``
* ``direct_comprehensive_output.mp4``
* ``direct_comprehensive_audio.mp3``
* ``s2s_comprehensive_output.mp3``
* ``s2s_comprehensive_latency_plot.png``
* ``lipsync_comprehensive_output.mp4``
* ``asd_comprehensive_output.csv``

Expected Results
~~~~~~~~~~~~~~~~

Successful test run:

.. code-block:: text

   ALL FUNCTIONAL TESTS PASSED!
   All clients are working correctly
   All services are communicating properly
   All outputs are being generated successfully

Failed test run will provide detailed error information for debugging.

Test Configuration
~~~~~~~~~~~~~~~~~~

**Timeouts:**

* Individual client tests: 5 minutes
* Complete test suite: 10 minutes per test
* Service health checks: 30 seconds

**Chunk Sizes:**

* Video chunks: 1MB (1048576 bytes) for basic tests
* Audio chunks: 2.0 seconds for basic tests

