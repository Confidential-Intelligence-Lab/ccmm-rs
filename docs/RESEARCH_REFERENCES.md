# Research foundations and references

FHE-rs is an independent implementation and research platform. This bibliography distinguishes established scientific foundations from our own implementation choices. References do **not** imply compatibility, benchmarking parity, or source-code reuse.

## Foundational homomorphic encryption and CKKS

1. C. Gentry, **Fully Homomorphic Encryption Using Ideal Lattices**, STOC, 2009. [DOI](https://doi.org/10.1145/1536414.1536440).
2. J. H. Cheon, A. Kim, M. Kim, Y. Song, **Homomorphic Encryption for Arithmetic of Approximate Numbers**, ASIACRYPT, 2017. [DOI](https://doi.org/10.1007/978-3-319-70694-8_15). Foundational CKKS approximate arithmetic.
3. J.-P. Bossuat, C. Mouchet, J. Troncoso-Pastoriza, J.-P. Hubaux, **Efficient Bootstrapping for Approximate Homomorphic Encryption with Non-sparse Keys**, EUROCRYPT, 2021. [DOI](https://doi.org/10.1007/978-3-030-77870-5_21). Background only: FHE-rs does not currently claim bootstrapping.

## RNS arithmetic, transforms, and implementation context

4. S. Halevi, V. Shoup, **Algorithms in HElib**, CRYPTO, 2014. [DOI](https://doi.org/10.1007/978-3-662-44371-2_31). Foundational efficient HE algorithms and implementation context.
5. J. H. Cheon, K. Han, A. Kim, M. Kim, Y. Song, **A Full RNS Variant of Approximate Homomorphic Encryption**, SAC, 2018 (proceedings published 2019). [DOI](https://doi.org/10.1007/978-3-030-10970-7_16). RNS-CKKS context.
6. Microsoft Research, **Microsoft SEAL**. [Repository](https://github.com/microsoft/SEAL). Reference implementation and user-oriented documentation structure, not an FHE-rs dependency.
7. **OpenFHE**. [Repository](https://github.com/openfheorg/openfhe-development). External HE library for ecosystem comparison and research context.
8. **HElib**. [Repository](https://github.com/homenc/HElib). External HE library for research context.

## FHE-rs specific research threads

The following are **research topics and implementation mechanisms**, not invented bibliographic citations. Before submitting the paper, add the exact paper titles, author lists, venues, years and DOIs that motivated each specific mechanism:

- **Park-style encrypted matrix multiplication / structured CCMM:** inspect the repository's Park characterization source and the previously selected CRYPTO research artifact; cite the verified primary paper, not a guessed attribution.
- **Grafting and composite-level CKKS:** connect the current RNS architecture to the particular Grafting paper(s) used by the implementation, with verified bibliographic identifiers.
- **Prepared public operands and NTT residency:** distinguish the project’s own R19/R30 validation evidence from external algorithmic prior art.
- **FFT2 decomposition and transpose avoidance:** cite the specific cleartext row/column, six-step or four-step FFT decomposition sources selected for the paper; do not conflate physical encrypted transpose avoidance with a general claim that all cleartext FFT algorithms avoid transposes.
- **HERACLES and related acceleration:** retain as hardware and architecture context only where independently verified.

For implementation discussion, see [eBLAS](EBLAS.md), [RNS architecture](RNS_ARCHITECTURE.md), and [FFT experiments](FFT_EXPERIMENTS_V3.md).

## Research and security claims

The executable benchmark evidence proves numerical output consistency for its stated configuration. It does not independently certify a specific security level, generalize a single workload to all input sizes, or show per-optimization causality. Production-security claims require comprehensive parameters and estimator analysis beyond the current large FFT2 validation.
