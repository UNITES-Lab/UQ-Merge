# UQ-Merge

## Environment preparation

To set up the environment, after creating a new conda environment, please follow
```bash
cd LLaVA
pip install -e .
cd ../lmms-eval
pip install -e .
cd ../mergekit
pip install -e .
```

## LLaVA folder
In `LLaVA` folder it contains the LLaVA model that is used in experiments.

## lmms-eval
`lmms-eval` folder contains the evaluation framework used to test all the models.

## mergekit
`mergekit` folder contains the implementation of all the merging methods.

## VLM-Uncertainty-Bench
`VLM-Uncertainty-Bench` has the uncertainty quantification code for UQ-Merge method.
