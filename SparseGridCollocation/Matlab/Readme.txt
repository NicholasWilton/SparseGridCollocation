a. In these folders, the main codes are cMuSIK.m (multilevel sparse collocation)
and cSIK.m (sparse collocation). File cfigure.m is to plot results and recall
cMuSIK.m and cSIK.m.

b. Each folder only could solve one particular PDE problem. The core codes in
different folders are almost same.

c. In these time dependent PDE problems, time is considered as one spatial
dimension.

d. It is avaible to run codes in 2Dheat, 3Dheat and BS on personal computer as
high levels approximations are commented.If more levels are released or to run
4Dheat, a supercomputer is necessary.

e. In the "Black Scholes" folder, firstly run the file To_get_ini.m in the folder
"MOL to find an earlier time" to get an estimation of option value at an 
earlier time than maturity time. Then copy the file Smoothinitial.mat to folder
"sparse grid collocation" to start sparse collocation.

Thank you very much for your reading!