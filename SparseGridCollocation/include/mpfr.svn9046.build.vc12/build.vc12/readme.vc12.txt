======================
A. A Note On Licensing
======================

Where files in this distribution have been derived from files licensed
under Gnu GPL or LGPL license terms, their headers have been preserved 
in order to ensure that these terms will continue to be honoured.  

Other files in this distribution that have been created by me for use
in building MPIR and MPFR using Microsoft Visual Studio 2012 are 
provided under the terms of the LGPL version 2.1

Running the MPFR tests automatically uses Python, which must hence be 
installed if you want to run them.  

=============================================
B. Compiling MPFR with the Visual Studio 2013
=============================================

The VC++ project files are intended for use with Visual Studio 
2013 Professional, but they can also be used with Visual C++ 2013 
Express to build win32 applications. 

Building MPFR
-------------

These VC++ build projects are based on MPIR 2.7 and MPFR-3.1. It
is assumed that MPIR has already been built and that the directories
containing MPIR and MPFR are at the same level in the directory 
structure:

    mpir
      dll         MPIR Dynamic Link Libraries 
      lib         MPIR Static Libraries
      build.vc12  Visual Studio 2013 build files
            ....
    mpfr
      dll         MPFR Dynamic Link Libraries
      lib         MPFR Static Libraries
      build.vc12  Visual Studio 2013 build files
            ....

The root directory name of the MPIR version that is to be used in 
building MPFR should be 'mpir' with any version number such as in
'mpir-3.1' removed.
 
The MPFR source distribution should be obtained and expanded into the
MPFR root directory (e.g. mpfr-3.1.0). After this the build project 
files should be added so that the build.vc12 sub-directory is in the
MPFR root directory as shown above.  After this the root directory 
should be renamed to 'mpfr'.

The root directory names 'mpir' and 'mpfr' are used because this makes 
it easier to use the latest version of MPIR and MPFR without having to 
update MPIR and MPFR library names and locations when new versions are 
released.
        
There are two build projects, one for static libraries and one for 
dynaimic link libraries:

    dll_mpfr.sln    for dynamic link libraries
    lib_mpfr.sln    for static libraries

After loading the appropriate solution file the Visual Studio IDE allows
the project configuration to be chosen:

    win32 or x64
    release or debug
    
after which the lib_mpfr library should be built first (but see Tuning
below), followed by lib_tests and then the tests.

If you wish to use the Intel compiler, you need to convert the build files
by right clicking on the MPFR top level Solution and then selecting the 
conversion option.

Any of the following projects and configurations can now be built:

    dll_mpfr    the DLL (uses the MPIR DLL) 
      Win32
        Debug
        Release
      x64
        Debug
        Release

    lib_mpfr    the static library (uses the MPIR static library) 
      Win32
        Debug
        Release
      x64
        Debug
        Release

After which the library output files are placed in the mpfr/lib or mpfr/dll
folderrs as appropriate.

Tuning
------

Because tuning is not reliable on Windows, tuning parameters are picked
up from the *nix builds. 

Before building MPFR, the choice of architecture for tuning should be
selected by editing the mparam.h file in the build.vc12 directory to
select the most appropriate tuning parameters.

Test Automation
----------------

Once the tests have been built the Python scripts run_dll_tests.py or
run_lib_tests.py found in the build.vc12 folder can be used to run the
tests (if Python is not installed the tests have to be run manually).

===================
C. Acknowledgements
===================

My thanks to:

1. The GMP team for their work on GMP and the MPFR team for their work 
   on MPFR
2. Patrick Pelissier, Vincent Lefèvre and Paul Zimmermann for helping
   to resolve VC++ issues in MPFR.
3. The MPIR team for their work on the MPIR fork of GMP.
4  Jeff Gilcrist for his help in testing, debugging and improving the
   readme.txt file giving the build instructions
 
       Brian Gladman, April 2014
