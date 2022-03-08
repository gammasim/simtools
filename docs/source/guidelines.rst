.. _Guidelines:

Guidelines for Developers
*************************

This section is meant for developers.

Testing
=======

pytest framework is used for unit testing.
The test modules are located in simtools/test.
Every module should have its respective test module and
ideally all functions should be covered by tests.

It is important to write the tests in parallel with the modules
to assure that the code is testable.

The pytest decorators mark.ignoreif are used to mark the tests that
requires: a) a config file properly set, b) a sim_telarray installation and
c) DB connection. Each of these are identified before each pytest session
and environment variables are used to store this information. See the implementation
in conftest.py. In util/tests.py one can find functions that reads these variables.
