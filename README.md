This is the repository associated with the paper *Multi-Output Conformal Regression: A Unified Comparative Study with New Conformity Scores*.

It includes:
- An implementation of several conformal methods for multi-output conformal regression.
- Several base predictors (Multivariate Quantile Function Forecaster, Distributional Random Forest, Gaussian Mixture parametrized by a hypernetwork).
- Metrics for marginal coverage, region size, and conditional coverage.
- A large empirical study based on datasets gathered from the literature, all with multiple outputs.

<p align="center">
<img src="images/unimodal_heteroscedastic.png?raw=true" alt="" width="95%" align="top">
<img src="images/taxi_example.png?raw=true" alt="" width="95%" align="top">
<img src="images/bimodal_heteroscedastic.png?raw=true" alt="" width="95%" align="top">
</p>

## Datasets

All datasets except MEPS are directly available in this repository. See step 5 of the installation for downloading MEPS.

Refer to these repositories for more information on the datasets used in this study:
- https://github.com/tsoumakas/mulan
- https://github.com/Shai128/mqr
- https://github.com/lorismichel/drf
- https://github.com/Zhendong-Wang/Probabilistic-Conformal-Prediction
- https://github.com/aschnuecker/Superlevel-sets

## Example usage

The following code shows an example usage of the code in this repository.

```python
from moc.configs.config import get_config
from moc.utils.run_config import RunConfig
from moc.models.mqf2.lightning_module import MQF2LightningModule
from moc.models.trainers.lightning_trainer import get_lightning_trainer
from moc.datamodules.real_datamodule import RealDataModule
from moc.metrics.metrics_computer import compute_coverage_indicator, compute_log_region_size
from moc.conformal.conformalizers import L_CP


config = get_config()
config.device = 'cpu'
rc = RunConfig(config, 'mulan', 'sf2')
datamodule = RealDataModule(rc)
p, q = datamodule.input_dim, datamodule.output_dim
model = MQF2LightningModule(p, q)
trainer = get_lightning_trainer(rc)
trainer.fit(model, datamodule)

alpha = 0.1
conformalizer = L_CP(datamodule.calib_dataloader(), model)
test_batch = next(iter(datamodule.test_dataloader()))
x, y = test_batch
coverage = compute_coverage_indicator(conformalizer, alpha, x, y)
volume = compute_log_region_size(conformalizer, model, alpha, x, n_samples=100)
print(coverage)
print(volume)
```

## Installation

### Prerequisites
- Python (tested on 3.10.14)

<!--
echo 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/' | sudo tee -a /etc/apt/sources.list
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
sudo apt update
apt policy r-base
sudo apt install r-base
-->

### Steps
1. Clone the repository:
```bash
git clone https://github.com/Vekteur/multi-output-conformal-regression.git
cd multi-output-conformal-regression
```

2. (Optional) Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```
for exact versions ensuring reproducibility, or
```
pip install -r requirements.in
```
for more flexibility.

4. (Optional) If you want to run Distributional Random Forests, install R (tested on 4.4.1, version 4.1 or higher is required).
Open the R interpreter using the command `R` and run the following command:
```R
install.packages("drf")
```
Compilation will take a few minutes.
Then run
```bash
pip install --index-url https://test.pypi.org/simple/ drf==0.1
```

5. (Optional) For running experiments on the MEPS dataset, download it according to [these instructions](https://github.com/yromano/cqr/tree/master/get_meps_data), summarized below:
```bash
git clone https://github.com/yromano/cqr
cd cqr/get_meps_data/
Rscript download_data.R
python main_clean_and_save_to_csv.py
cd ../../
for id in 19 20 21; do mv "cqr/get_meps_data/meps_${id}_reg.csv" "data/feldman/meps_${id}.csv"; done
rm -rf cqr
```

## Reproducing the results

To generate the figures for toy datasets, run `toy_experiments.ipynb`.

To compute the results of the paper:
```
python run.py name="full" device="cuda" repeat_tuning=10
```
or use `device="cpu"` if you don't have a GPU.

To generate the figures based on these results, run `analysis.ipynb` 
