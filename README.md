# Hierarchical Interpolation with Projection, Evaluation and Derivation

The purpose of this code is to define arbitrary interpolation trees with some utilities (evaluation, projection and derivation) that can be used for instance in multimaterial topology optimization algorithms [^1][^2] as well as in other applications.

This code is a major upgrade of [^3], based itself on [^4].


## Quickstart


Start by running the example files in the folder "examples". If your Matlab version supports it, you can run the LiveScripts (in "examples/notebooks"), or traditional .m files (in "examples/mfiles"), in the alphabetical order.


Some nice pictures

**NB** The code has been tested with success on Matlab R2023a.


## Contents
examples
    | mfiles
    |    | a_SimpleDomains.m : examples of simple interpolation domains
    |    | b_SimpleScalarInterp.m : interpolation of scalar functions on a simple domain.
    |    | c_SimpleVectorInterp.m : interpolation of vector functions on a simple domain.
    |    | d_HierarchicalScalarInterp.m : interpolation of scalar functions on hierarchized domains.
    |    | e_HierarchicalVectorInterp.m : interpolation of vector functions on hierarchized domains.
    |    | f_VertexFunctionOperations.m : some playaround with the functions
    | notebooks 
         |(same as in mfiles, but .mlx format)
src
    | class (main code is here, definition of the classes)
        | Domain.m
        | Interpolation.m
        | Penalization.m
        | ShapeFunction.m
        | VertexFunction.m
    | others (auxiliary functions for computation)
        | mult.m (function to simplify pagewise multiplications)
        | t.m (just a shorthand for pagetranspose)
    | vizualization (auxiliary functions for vizualization)
        | getTransformedPlan.m
        | legend2color.m
        | plotcolor2D.m
        | rot3D.m
        | rotPoint.m
LICENSE
README.md

## Citation

Please use the following citation reference if you use the code:

    T. Cherrière tcherrie/hiped: Hiped (v1.0.1), March 2024. Zenodo.

Bibtex entry :



**NB: version number and DOI must be adapted from [Zenodo's repository.](https://doi.org/10.5281/zenodo.7701776)** 

## License

Copyright (C) 2024 Théodore CHERRIERE (theodore.cherriere@ricam.oeaw.ac.at)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.

## References

[^1] Cherrière, T., Laurent, L., Hlioui, S. et al., (2022). 
"Multi-material topology optimization using Wachspress interpolations for 
designing a 3-phase electrical machine stator."
Structural and Multidisciplinary Optimization, 65(352). 
https://doi.org/10.1007/s00158-022-03460-1

[^2] Cherrière, T. (2024) "Toward topology optimization of hybrid-excited 
electrical machines using recursive material interpolation"
SCEE 2024, Darmstadt, Germany. 

[^3] Cherrière, T. and Laurent, L. (2023), Wachspress2D3D, 
https://doi.org/10.5281/zenodo.7701776
 
[^4]: M. Floater, A. Gillette, and N. Sukumar, (2014).
“Gradient bounds for Wachspress coordinates on polytopes,”
SIAM J. Numer. Anal., vol. 52, no. 1, pp. 515–532, 
doi: 10.1137/130925712
