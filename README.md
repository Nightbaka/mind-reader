# mind-reader

Experiments on dataset [A Polish Electroencephalography, Alzheimer’s Risk-genes, Lifestyle and Neuroimaging (PEARL-Neuro) Database](https://openneuro.org/datasets/ds004796). The goal is to explore the use of Variational Autoencoders (VAE) and Diffusion Models for EEG data reconstruction and generation.

## Reproduction Instructions

1. Install dependencies:  
   ```sh
   pip install -r requirements.txt
   ```

2. Download the raw Sternberg EEG data:  
   ```sh
   aws s3 cp \
     --no-sign-request \
     --recursive \
     --exclude "*" \
     --include "*task-sternberg_eeg*" \
     --include "*task-sternberg_events*" \
     s3://openneuro.org/ds004796/ \
     data/raw/
   ```

3. Preprocess the data (filter, resample, epoch):  
   ```sh
   python src/preprocess.py
   ```

4. Run the VAE experiments notebook:  
   ```sh
   jupyter notebook experiments/vaeeg_experiments.ipynb
   ```

5. Run the diffusion model experiments notebook:  
   ```sh
   jupyter notebook experiments/diffe_experiments.ipynb
   ```