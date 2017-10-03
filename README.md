# SparseGridCollocation

Theis repository is quite large but download, clone with git or checkout with subversion

## Whats included
SparseGridCollocationCpp - pure C++ version with no dependency on CUDA or an NVidia Graphics card

SparseGridCollocation - The main CUDA-enabled project

Dissertation - Pdf, tex and supporting files for the dissertation document.

SparseGridCollocation\MatLab - YangZhang's original MatLab code with some small additions

SparseGridCollocation\Documents\Quick Start Guide.docx - Some helpful advice on building the code in Visual Studio


## Dependencies
Eigen - should be already within the \include directory

QuantLib - download and install the version listed in the dissertation document, drop the source code in the \includes directory to re-point the C++ include path in Visual Studio project properties

Intel MKL - as above

CUDA SDK 8 - install if running the CUDA enabled version.


## Warning - DO NOT upgrade CudaLib or ThrustLib projects when opening in Visual Studio 2017. CUDA SDK does not yet support this version of Visual Studio project file format so will break.

Whilst I've tested both solutions on several platforms, if anything appears to be missing please don't hesitate to contact me.

## Warning - The UnitTest project will not currently build, but it is not needed to compile MuSiK-c nor to run the TestHarness or the Experiements. 
All others in both solutions will build if all dependencies are installed correctly.