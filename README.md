# chloroDAG

This repository contains the code and materials accompanying the research article:

> **Causal inference of post-transcriptional regulation timelines from long-read sequencing in _Arabidopsis thaliana_**

We propose a novel framework for reconstructing the chronology of genetic regulation using causal inference based on Pearl’s theory. The approach proceeds in three main stages: **causal discovery**, **causal inference**, and **chronology construction**. We apply it to the *ndhB* and *ndhD* genes of the chloroplast in _Arabidopsis thaliana_, generating four alternative maturation timeline models per gene using different causal discovery algorithms (**HC**, **PC**, **LiNGAM**, and **NOTEARS**).

The framework addresses two key methodological challenges:

- Handling missing data using an **EM algorithm** that jointly imputes missing values and estimates the Bayesian network.
- Selecting the **ℓ₁-regularization parameter** in NOTEARS via a **stability selection** strategy.

The resulting causal models consistently outperform reference chronologies in terms of reliability and model fit. Furthermore, the integration of causal reasoning with domain expertise enables the formulation of testable biological hypotheses and the design of targeted experimental interventions.

## 🧬 Data

The project analyzes long-read sequencing data for the chloroplast genes *ndhB* and *ndhD* in _Arabidopsis thaliana_ based on [*Guilcher et al*](https://doi.org/10.3390/ijms222011297).

## 🧪 Key Contributions

- A reproducible pipeline for causal reconstruction of RNA maturation timelines.
- Robust handling of missing data via joint imputation and model estimation.
- Stability-based hyperparameter selection for NOTEARS.
- Application to *Arabidopsis thaliana* chloroplast genes (*ndhB*, *ndhD*).
- Reproducible scientific publishing.


## 📁 Repository Structure

chloroDAG/

        - Code/ # Python source code for causal modeling and analysis

        - Data/ # Input datasets

        - Paper/ # Quarto source (.qmd), rendered PDF and LaTeX files

        - README.md # Project overview (this file)

        - requirements.txt # Python dependencies

        - Results/ # Output DAGs, charts, visualisations, evaluation reports

## 👤 Authors

- [Rubén Martos](https://orcid.org/0000-0002-1463-5088). Université d'Évry Paris-Saclay (LaMME)

- [Christophe Ambroise](https://orcid.org/0000-0002-8148-0346). Université d'Évry Paris-Saclay (LaMME)

- [Guillem Rigaill](https://orcid.org/0000-0002-7176-7511). Université Paris-Saclay, CNRS, INRAE, Université d'Évry (IPS2, LaMME)

<!-- ## 📚 Citation

If you use this work, please cite:

> Causal inference of post-transcriptional regulation timelines from long-read sequencing in Arabidopsis thaliana
[DOI or preprint link, si aplica] -->
