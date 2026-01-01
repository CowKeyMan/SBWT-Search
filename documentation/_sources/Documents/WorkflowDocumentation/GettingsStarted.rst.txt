Getting Started
===============

Setup
+++++

For setting up, it is recommended to first read the :ref:`Tools` page then come back to this page.

Typical workflow
++++++++++++++++

This section will describe a typical workflow from writing a new piece of code, to checking it against formats and testing it, to seeing code coverage and documentation for it. To understand what each script does, it is recommended to check the :ref:`File Documentation` page and read their more detailed description there. If that does not suffice, you can also open up each script in a text editor and see what they do.

#. Writing a new module

  #. Write a new header file template
  #. Write a new cpp file for it
  #. Go to `src/BuildCommon.cmake` and add the new library there

#. Testing a new module or a change

  #. Write the test module in the same folder, see the :ref:`Practices` section for more details on how the test files should be named and where they should be placed
  #. Go to `src/BuildTests.cmake` and add the new test file there
  #. Run `./scripts/build/tests.sh <platform>` to build the testing executable
  #. Run `./scripts/test/test.sh` to run the tests and generate the code coverage report which can be found by opening `docs/index.html` in a browser and choose the *Code Coverage* option

#. Debugging the change

   #. If you wish, you may debug with the test executable
   #. To debug with the main program, you may run `./scripts/build/debug.sh <platform>` and the main executable will be found in *build/bin/sbwt_search*

#. Releasing

  #. Generate a release build which can be used for timing with `./scripts/build/release.sh <platform>` to generate a build of your code.

#. You can run your code by running `./build/bin/sbwt_search`, which is the main executable generated. Instructions on running will be given when you run the progam. The same instructions are also given in the README.

# Updating the doumentation

  # The documentation is partially generated from the codebase, especially the :ref:`File Documentation` page, and the other part is made up of `Sphinx reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_ files in the `documentation/Documents` folder. Note that the comments on top of our source files are also interpreted as reStructuredText when they are output to the :ref:`File Documentation` page.
  #. After updating the documentation, run `./scripts/build/docs.sh`, which will generate the documentation in the docs folder. Similarly with the code coverage, you may open *docs/index.html* in a browser and this time choose the *Documentation* option.

#. Pushing to github

   #. To push, please first run `git config core.hooksPath hooks`. This will run the tests on the codebase to ensure proper styling and makes sure that the tests pass.
   #. If you get a complaint from clang-format, you can run `./scripts/modifiers/apply_clang_format.py`.
