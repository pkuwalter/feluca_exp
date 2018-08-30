# feluca_exp

This is an iteration version of Feluca on multi-GPUs, the problem for this version is that there are too manu colors are needed.
For the amazon dataset, there are about 5000 colors are needed. This problem is caused by the color exact operation on GPU. 
Please see the function kernel_extract_color.