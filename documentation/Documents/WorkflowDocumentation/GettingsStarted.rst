Getting Started
===============

Setup
+++++

For setting up, it is recommended to first read the :ref:`Tools` page then come back to this page.

Typical workflow
++++++++++++++++

This section will describe a typical workflow from writing a new piece of code, to checking it agianst formats and testing it, to seeing code coverage and documentation for it. To understand what each script does, it is recommended to open them up in a text editor and seeing what they do.

#. Write a new header file template
#. Write a new cpp file for it
#. Write a test for it to test that the new functions work
#. Run `./scripts/build/build_test.sh` to build the testing executable
#. If clang-tidy or clang-format say that there is some misformatted code or if there are errors in the build, then fix them and run again.

   #. At this point you may want to consider running `./scripts/standalone/apply_clang_format.py` if you agree with the changes proposed by clang-format

#. Run `./scripts/standalone/run_tests.sh` to run the tests and generate the code coverage report which can be found by opening *docs/index.html* in a browser and choose the *Code Coverage* option
#. Integrate the new functions into the program and test again
#. Once you have a main executable ready to be run you can run `./scripts/build/build_all.sh` to generate a debug build of your code.
#. You can run your code by running `./build/src/main`, which is the main executable generated
#. Here you may update the documentation. You can run `./scripts/build/build_docs.sh`, which will generate the documentation in the docs folder. Similarly with the code coverage, you may open *docs/index.html* in a browser and this time choose the *Documentation* option.
#. After you are satisfied and wish to generate a release executable, you may run `./scripts/build/build_release.sh` and then you may execute it by running `./build/src/main`.
