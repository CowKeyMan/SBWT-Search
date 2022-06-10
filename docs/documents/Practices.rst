Practices
=========

Certain practices while developing the code should be respected among developers. This section is dedicated to defining the most prominent ones

Trailing return type
++++++++++++++++++++

Trailing return type is the new way in C++11 to declare functions. It is great because it ensures consistency of style between function declarations and variable declarations. (It also resembles python a lot!)

As an example, whereas traditionally we write the following: ``float add(int a, int b) { ... }``, we now write: ``auto add(int a, int b) -> float { ... }``

As of C++14, we can also just write ``auto add(int a, int b) { ... }``, and the return type will be deduced automatically, however, for now, we keep the return type in the function for the sake of the programmers so we know what we are returning.
