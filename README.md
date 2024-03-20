# Hierarchical Interpolation with Projection, Evaluation and Derivation

[![GitHub license](https://img.shields.io/github/license/tcherrie/Hiped)](https://github.com/tcherrie/Hiped) [![GitHub release](https://img.shields.io/github/release/tcherrie/Hiped.svg)](https://github.com/tcherrie/Hiped/releases/) [![GitHub stars](https://img.shields.io/github/stars/tcherrie/Hiped)](https://github.com/tcherrie/Hiped/stargazers)
[![DOI](https://zenodo.org/badge/763998095.svg)](https://zenodo.org/doi/10.5281/zenodo.10718117)



The purpose of this code is to define arbitrary interpolation trees with some utilities (evaluation, projection and derivation) that can be used for instance in multimaterial topology optimization algorithms [^1][^2] as well as in other applications using projected gradient descent.

This code is a major upgrade of [^3], based itself on [^4].


## Quickstart


Start by running the example files in the folder "examples". If your Matlab version supports it, you can run the LiveScripts (in "examples/notebooks"), or traditional .m files (in "examples/mfiles"), in alphabetical order.

**NB** : The code has been tested with success on Matlab R2023a.


## Purpose
Let $a\in \mathbb{R}^d$, and $x \in \mathcal{D}$ a convex polytope in 1,2 or 3 dimensions. The polytope has $n$ vertices and looks like one on the following figure :

![image](https://github.com/tcherrie/Hiped/assets/72595712/164ed134-ef6e-4506-9e88-63bb7bc9d1d0)


This repository contains a general framework to define a mapping $m : \mathcal{D} \times \mathbb{R}^d \rightarrow  \mathbb{R}^d$, which define an interpolation between $n$ functions $f_i : a \rightarrow f_i(a)$ on $\mathcal{D}$.

$$ m(a,x) = \sum_{i}^n  \omega_i(x) f_i(a),$$

each of the $f_i$ being associated to a vertex of $\mathcal{D}$, and each of the weights $\omega_i$ being a generalized barycentric coordinate of $\mathcal{D}$ [^4]. This code can also handle penalizations $P_i$ :

$$ m(a,x) = \sum_{i}^n  P_i(\omega_i(x)) f_i(a),$$

and extend the interpolation domain recursively, by defining a rooted tree of interpolations :

$$ m(a,x) = \sum_{i}^n  P_i(\omega_i(x)) m_i(a,x)$$

An illustration of such a hierarchical interpolation domain is given in the following figure.

![image](https://github.com/tcherrie/Hiped/assets/72595712/4a8f4956-98c0-4d8e-9e71-89ba0de4b022)

The routines computing the evaluation of the interpolation, the derivatives, and the projection of exterior $x$ onto $\mathcal{D}$ are provided, making it suitable for a projected gradient descent in optimization.

## Contents
- examples
    - mfiles   
        - a_SimpleDomains.m : examples of simple interpolation domains
        - b_SimpleScalarInterp.m : interpolation of scalar functions on a simple domain.
        - c_SimpleVectorInterp.m : interpolation of vector functions on a simple domain.
        - d_HierarchicalScalarInterp.m : interpolation of scalar functions on hierarchized domains.
        - e_HierarchicalVectorInterp.m : interpolation of vector functions on hierarchized domains.
        - f_VertexFunctionOperations.m : some playaround with the functions
    - notebooks (same as in mfiles, but .mlx format)
- src
    - class (main code is here, definition of the classes)
        - Domain.m
        - Interpolation.m
        - Penalization.m
        - ShapeFunction.m
        - VertexFunction.m
    
    - others (auxiliary functions for computation)
        - mult.m (function to simplify pagewise multiplications)
        - t.m (just a shorthand for pagetranspose)
    - vizualization (auxiliary functions for vizualization)
        - getTransformedPlan.m
        - legend2color.m
        - plotcolor2D.m
        - rot3D.m
        - rotPoint.m
- LICENSE 
- README.md

## Citation

Please use the following citation reference if you use the code:

    T. Cherrière. tcherrie/Hiped: Hierarchical Interpolation with Projection, Evaluation and Derivation (v1.0.1), February 2024. Zenodo. https://doi.org/10.5281/zenodo.10718117

Bibtex entry:

    @software{tcherrie_2024_10718117,
    author       = {Cherrière, Théodore},
    title        = {tcherrie/Hiped: Hierarchical Interpolation with Projection, Evaluation and Derivation},
    month        = feb,
    year         = 2024,
    publisher    = {Zenodo},
    version      = {v0.0.2},
    doi          = {10.5281/zenodo.10718117},
    url          = {https://doi.org/10.5281/zenodo.10718117}
    }

**NB: version number and DOI must be adapted from [Zenodo's repository](https://doi.org/10.5281/zenodo.10718117)**, according to the version used.

## License

Copyright (C) 2024 Théodore CHERRIERE (theodore.cherriere@ricam.oeaw.ac.at)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.

## References

[^1]: Cherrière, T., Laurent, L., Hlioui, S. et al., (2022). 
"Multi-material topology optimization using Wachspress interpolations for 
designing a 3-phase electrical machine stator."
Structural and Multidisciplinary Optimization, 65(352). 
https://doi.org/10.1007/s00158-022-03460-1

[^2]: Cherrière, T. (2024) "Toward topology optimization of hybrid-excited 
electrical machines using recursive material interpolation",
SCEE 2024, Darmstadt, Germany. 

[^3]: Cherrière, T. and Laurent, L. (2023), Wachspress2D3D, https://doi.org/10.5281/zenodo.7701776
 
[^4]: M. Floater, A. Gillette, and N. Sukumar, (2014).
“Gradient bounds for Wachspress coordinates on polytopes,”
SIAM J. Numer. Anal., vol. 52, no. 1, pp. 515–532, 
doi: 10.1137/130925712
