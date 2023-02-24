Tools
=====

The project was setup in a way to use good C++ practices. Here is discussed the tools which enforce those practices. Some of the tools use python, as such it is recommended to create a new python 3 environment for this workflow template.

CMake
+++++

This is the tool used for compiling the project and documentation. However, besides just building, it also runs check scripts for formatting, etc. Each *CmakeLists.txt* or *.cmake* file has a description at the top as its description as documentation, thus we will not be going through each script here.

Installation
------------

.. code-block:: bash

  sudo apt install build-essential

We also use ccache as a tool with cmake, to cache results of previous compilations and make future compilations run faster, so it is probably worth installing that as well.

.. code-block:: bash

  sudo apt install ccache

Doxygen
+++++++

This is the tool used to generate automated documentation from the source files.

Installation
------------

Make sure that you install the correct version of Doxygen. At the time of writing, this repository is using Doxygen 1.9.2. The version can be found at the first line of docs/Doxyfile.in.

The easy way is to install using:

.. code-block:: bash

  sudo apt install doxygen

However, to install the very latest doxygen, you may need to go through the (simple) installation here: https://www.doxygen.nl/download.html. Here are the installation commands:


.. code-block:: bash

  sudo apt install flex bison
  git clone https://github.com/doxygen/doxygen.git
  cd doxygen
  mkdir build
  cd build
  cmake -G "Unix Makefiles" ..
  make
  sudo make install
  cd ../..
  rm -r build

You may encounter some missing dependencies when building doxygen (such as flex and bison above), in which case please install these in the same way we are installing flex and bison above.

Sphinx
++++++

Sphinx is a layer over Doxygen to generate more beautiful documentation. It reads the doxygen outputs and incorporates them into its own documentation. It also allows us to easily add more pages to the documentation, such as this page itself! We can use latex easily as well within our documentation with this tool. Traditionally it is used for Python documentation.

We use the some extensions with Sphinx:
  * sphinx-rtd-theme: read the docs theme is used since it is great for documentation.
  * sphinx-rtd-dark-mode: dark mode for users which prefer this. It is easily togglable in the webpage
  * breathe: For reading doxygen output and parsing it
  * exhale: For outputting the doxygen content into Sphinx format

Installation
------------

.. code-block:: bash

  pip install sphinx
  pip install breathe
  pip install exhale
  pip install sphinx-rtd-dark-mode
  pip install sphinx-rtd-theme

Mermaid
+++++++

For diagrams in the docentation we use Mermaid, which is a text based graph creator. This needs to be installed separately.


Installation
------------

You may need an updated version of nodejs for this to work, as such, you can uninstall the current version you have using these commands

.. code-block:: bash

  sudo apt remove npm
  sudo apt remove nodejs
  sudo apt autoremove

Then install a newer version of nodejs (in this case version 18) using the following command:

.. code-block:: bash

  cd ~
  curl -sL https://deb.nodesource.com/setup_18.x -o nodesource_setup.sh
  sudo bash nodesource_setup.sh
  rm nodesource_setup.sh

Then you can install mermaid using the following:

.. code-block:: bash

  pip install sphinxcontrib-mermaid
  npm install @mermaid-js/mermaid-cli


clang-format
++++++++++++

This tool is used for ensuring a consistent language format across developers. This includes using spaces vs tabs, how many tabs or spaces, if a line should be skipped after or before an opening curly brackets, etc.

**Official Documentation for Options**: https://clang.llvm.org/docs/ClangFormatStyleOptions.html

At the time of writing, this repository is using version 14 of clang-format

Installation
------------

Go to https://github.com/llvm/llvm-project/releases and download the latest version for your system. Note, the latest version might not have a build for your system, so go look for ones which do have a version for your system.

.. code-block:: bash

  cd /opt # This is the folder where we will put the executables
  sudo wget https://github.com/llvm/llvm-project/releases/download/llvmorg-14.0.0/clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz -O clang+llvm.tar.xz # download your version (make sure to change the link!) and save it as a file named clang+llvm.tar.xz
  sudo tar xf clang+llvm.tar.xz # extract it
  sudo printf "\n#add clang to path\nexport PATH=\"/opt/clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04/bin:\${PATH}\"" >> ~/.bashrc # Add to PATH (make sure to change the folder name version!)

googletest
++++++++++

The testing framework used is the popular googletest which can test both C and C++ code.

Installation
------------

Installation will be automatic when you run cmake as it will be done using FetchContect.

lcov
++++

This is the tool to display code coverage after running googletest. This tool was very difficult to set up on native Windows, hence for any Windows users, it is recommended to use WSL for code coverage, or else remove the parts of the scripts which use it (found in `scripts/standalone/run_tests.sh`)

Unfortunately lcov does not capture CUDA `__device__` functions, hence we put these functions in separate *.cuh* files.

Installation
------------

.. code-block:: bash

  sudo apt install lcov

Github pages
++++++++++++

We use github pages to publish the documentation and code coverage. To set this up, we must have a separate branch called *gh-pages*. Then go to github, and in your repository's settings you can find the settings for the pages. Set this up so that it uses the root folder of your gh-pages branch.

CUDA
++++

This template also supports CUDA. Check the official installation guides for more info, which can be found here: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html. Note: WSL and Windows have their own guides. If you do not wish to use CUDA, you can turn the build off. For more information look at the **build** folder section in :ref:`Workflow Files`.

On github actions we do not have GPUs, but we can still compile and test our code. Thus, our gpu tests will fail but we will still have the code coverage report on the site.

kseqpp_REad
+++++++++++

kseq++ is a program to read FASTA/Q files. Its repository is at https://github.com/cartoonist/kseqpp. We use another version of it which is kseqpp_Read whose repository is at: https://github.com/CowKeyMan/kseqpp_REad. However, it may neeed a dependency which is zlib.

Installation
------------

.. code-block:: bash

  sudo apt install zlib1g-dev # a dependency

spdlog
++++++

This is our logging tool. It is installed automatically by CMake. To see the logs when running the main program, simply run the below code and you will see a lot of logs. This is used for timing our code as well.

.. code-block:: bash

 export SPDLOG_LEVEL=TRACE
