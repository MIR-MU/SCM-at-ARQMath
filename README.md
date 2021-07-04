# Soft Cosine Measure at ARQMath

This repository contains the math information retrieval (MIR) system for the
[ARQMath][] competition based on the tf-idf model and the soft cosine measure
[[1][]]:

 [Joint word embeddings and soft cosine measure at ARQMath][pv173-talk]

The system build on the following other repositories:

- [ARQMath-data-preprocessing][]: Scripts for producting preprocessed
  [ARQMath][] competition datasets.
- [ARQMath-eval][]: Python package for evaluating the performance of a MIR
  system on a number of tasks, including [ARQMath][].

 [arqmath]: https://www.cs.rit.edu/~dprl/ARQMath/
 [arqmath-data-preprocessing]: https://github.com/MIR-MU/ARQMath-data-preprocessing
 [arqmath-eval]: https://github.com/MIR-MU/ARQMath-eval
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

## Citing
### Text

NOVOTNÝ, Vít, Petr SOJKA, Michal ŠTEFÁNIK and Dávid LUPTÁK. Three is Better
than One: Ensembling Math Information Retrieval Systems. *CEUR Workshop
Proceedings*. Thessaloniki, Greece: M. Jeusfeld c/o Redaktion Sun SITE,
Informatik V, RWTH Aachen., 2020, vol. 2020, No 2696, p. 1-30. ISSN 1613-0073.

### BibTeX
``` bib
@inproceedings{mir:mirmuARQMath2020,
  title = {{Three is Better than One}},
  author = {V\'{i}t Novotn\'{y} and Petr Sojka and Michal \v{S}tef\'{a}nik and D\'{a}vid Lupt\'{a}k},
  booktitle = {CEUR Workshop Proceedings: ARQMath task at CLEF conference},
  publisher = {CEUR-WS},
  address = {Thessaloniki, Greece},
  date = {22--25 September, 2020},
  year = 2020,
  volume = 2696,
  pages = {1--30},
  url = {http://ceur-ws.org/Vol-2696/paper_235.pdf},
}
```
