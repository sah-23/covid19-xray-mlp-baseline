## COVID-19 Chest X-ray Classifier

Simple Keras/TF classifier that distinguishes COVID vs Non-COVID from chest X-rays using a multilayer perceptron (MLP) on flattened 100×100 images. The workflow is contained in the `Covid_checker.ipynb` notebook.

### Dataset

- Source: Kaggle — COVID-19 Radiography Database (`COVID` and `Viral Pneumonia`/`Normal` subsets) by Pranav Rai Kokte. [Dataset link](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)
- Download from Kaggle and extract locally. The notebook expects a folder layout like:

```
dataset_root/
  train/
    COVID/
      img1.png
      ...
    NonCOVID/
      img2.png
      ...
```

If your extracted dataset uses class names `COVID` and `NORMAL`, rename `NORMAL` to `NonCOVID` (or adjust the class mapping in the notebook accordingly).

In the notebook, set the Windows path to the training images glob, for example:

```
address = r"C:\\path\\to\\dataset_root\\train\\*"
```

### Quickstart

1) Create environment and install deps (Windows bash or PowerShell):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2) Launch Jupyter and open `Covid_checker.ipynb` (VS Code, JupyterLab, or classic Jupyter). Set `address` as shown above and run the cells top-to-bottom.

### Model at a glance

- Input: 100×100 RGB image flattened to 30,000 features
- Architecture: Dense layers with ReLU activations; final 2-unit output layer
- Loss/metrics: `binary_crossentropy`, `accuracy`

### Tips and improvements

- The MLP discards spatial information. A small CNN will likely outperform this baseline.
- Consider using a validation split, early stopping, and class weights if classes are imbalanced.
- For multi-class setups, prefer a `softmax` head with `categorical_crossentropy`.

### Project structure

- `Covid_checker.ipynb`: Data loading, preprocessing, training, and evaluation
- `requirements.txt`: Python dependencies

### License

Add a license (e.g., MIT) if making this public. Ensure your dataset usage complies with the Kaggle dataset license/terms.

