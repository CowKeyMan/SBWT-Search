Practices
=========

..
   Change these practices as you see fits your project

Certain practices while developing the code should be respected among developers. This section is dedicated to defining the most prominent ones

Trailing return type
++++++++++++++++++++

Trailing return type is the new way in C++11 to declare functions. It is great because it ensures consistency of style between function declarations and variable declarations. (It also resembles python a lot!)

As an example, whereas traditionally we write the following: ``float add(int a, int b) { ... }``, we now write: ``auto add(int a, int b) -> float { ... }``

As of C++14, we can also just write ``auto add(int a, int b) { ... }``, and the return type will be deduced automatically, however, for now, we keep the return type in the function for the sake of the programmers so we know what we are returning.

Naming Conventions
++++++++++++++++++

File naming convention
----------------------

When naming files, we name the related header, source and test files with the same prefix. This way when we sort by name we can see them grouped together. For example, if our *Functions.cpp* file has a corresponding *Functions.h*, then our test file will be called *Functions_test.cpp* (as opposed to *Test_Functions.cpp*, which would end up grouping the tests together instead). File names use CamelCase, and then we separate tags (tags such as *test*) with underscores. These are valid file names: File1.cpp, File2.h, FileThree.cpp, FileThree_test.cpp.

Function and variable names
---------------------------

This repository uses snake_case for all variables and function names. For example, these are valid function and variable names: variable, my_variable, and_this_is_a_function, my2_variable.

Namespaces
----------

With regards to namespaces, snake_case is also used.

Header Guards
+++++++++++++

Each header file should be in the following format:

.. code-block:: cpp

    #ifndef FILE_NAME_H
    #define FILE_NAME_H
    // everything else here
    #endif

Above, *FILE_NAME* will be the name of the file but we switch from SnakeCase to ALL_CAPS_SNAKE_CASE and we add the *_H* at the end. So a file named 'HelloWorld_test.' should have a header guard defining 'HELLO_WORLD_TEST_H'

In-code comments
++++++++++++++++

To keep the codebase clean of redundant comments, we try to use clear naming conventions to beging with. When something needs to be non-clean, only then do we put a comment next to it.

Furthermore, every header file should have a comment after the header guards, defining what it does. Look at an existing header file for the correct format, as they all should follow that structure to be recognised by doxygen.

Testing
+++++++

* Executables with main functions are not tested
* Prefer constructors and destructors over `virtual void SetUp()` and `virtual void TearDown()`. The latter may be necessary in some cases such as when inheritance is needed or when the destructor code can throw exceptions.
