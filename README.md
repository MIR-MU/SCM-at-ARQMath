# Soft Cosine Measure at ARQMath

This repository contains the math information retrieval (MIR) system for the
[ARQMath][] competition based on the tf-idf model and the soft cosine measure
[[1][]]:

 ![Joint word embeddings and soft cosine measure at ARQMath][pv173-talk]

The system build on the following other repositories:

- [ARQMath-data-preprocessing][]: Scripts for producting preprocessed
  [ARQMath][] competition datasets.
- [ARQMath-eval][]: Python package for evaluating the performance of a MIR
  system on a number of tasks, including [ARQMath][].

 [arqmath]: https://www.cs.rit.edu/~dprl/ARQMath/
 [arqmath-data-preprocessing]: https://gitlab.fi.muni.cz/xnovot32/arqmath-data-preprocessing
 [arqmath-eval]: https://gitlab.fi.muni.cz/xstefan3/arqmath-eval/-/tree/master/task1-votes/xnovot32
 [pv173-talk]: https://nlp.fi.muni.cz/trac/research/chrome/site/seminar2020/scm-at-arqmath.mp4

## Usage

To set up our system, execute the following commands:

```sh
git submodule update --init
pip install -r input_data/requirements.txt
pip install -r requirements.txt
```

Then, you can either:

1. download our results by executing the `dvc pull` command, or
2. reproduce our results by installing the dependencies of the
   [ARQMath-data-preprocessing][] repository, reproducing or downloading the
   results in the [ARQMath-data-preprocessing][]  repository, and then
   executing the `dvc repro` command.

## Bibliography

1. NOVOTNÝ, Vít. Implementation Notes for the Soft Cosine Measure. In
   *Proceedings of the 27th ACM International Conference on Information and
   Knowledge Management (CIKM '18)*. Torino, Italy: Association for Computing
   Machinery, 2018. s. 1639-1642, 4 s. ISBN 978-1-4503-6014-2.
   doi:[10.1145/3269206.3269317][1].

 [1]: https://doi.org/10.1145/3269206.3269317
