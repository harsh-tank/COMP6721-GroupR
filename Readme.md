# Satellite Image Classification using CNN.

Traditional land cover classification methods are labor-intensive and time-consuming, relying on field surveys and manual interpretation of aerial photographs. However, satellite imagery offers extensive spatial coverage and spectral data, making it ideal for diverse applications like environmental monitoring and urban planning.

Our project leverages Convolutional Neural Network (CNN) models to accurately classify land cover using satellite images. Challenges include preprocessing raw data, handling low-quality images, and combating overfitting through techniques like data augmentation and controlling learning rate parameters. Evaluation metrics such as precision, recall, and F1 score, along with visualization tools like confusion matrices and ROC curves, are used to assess model performance.

## Datasets

Following are the details and download links of our datasets.
It should be downloaded in local machine in order to run our models.

| Dataset              | Author                                                                                        | Download Links                                                                                      |
| -------------------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Sat Data (RSI-CB256) | [mahmoudreda55](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification) | [Kaggle](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification)              |
| EuroSAT              | [nilesh789](https://www.kaggle.com/nilesh789)                                                 | [Kaggle](https://www.kaggle.com/code/nilesh789/land-cover-classification-with-eurosat-dataset/data) |
| NWPU Dataset         | [Planet](https://gjy3035.github.io/NWPU-Crowd-Sample-Code/)                                   | [Kaggle](https://www.kaggle.com/datasets/happyyang/nwpu-data-set)                                   |

## Requirements

The following bullet points are the links of the libraries/frameworks we used in our project. In order to run our models,
these libraries must be installed in the local machine.

- [Python3](https://www.python.org/downloads/)
- [Pytorch](https://pytorch.org/)
- [Numpy](https://numpy.org/install/)
- [OpenCV](https://opencv.org/releases/)
- [Matplotlib](https://matplotlib.org/stable/users/installing/index.html)
- [SciPy](https://scipy.org/install/)
- [Sklearn](https://scikit-learn.org/stable/install.html)
- [Optuna](https://optuna.org/#installation)
- [Pandas](https://pandas.pydata.org/docs/getting_started/install.html)
- [Seaborn](https://seaborn.pydata.org/installing.html)

## How to train model:

1. Clone the repository using git clone <repository_url> to download the project files locally.
2. Install necessary dependencies by navigating to the project directory and running pip install -r requirements.txt.
3. If the model weights are not included, download them from the provided link and place them in the appropriate directory.
4. Launch Jupyter Notebook and open the notebook file (your_notebook.ipynb) containing the model code.
5. Run the notebook cells to train the model, ensuring validation steps are included to monitor performance.
6. Locate the evaluation section in the notebook to view metrics like accuracy, precision, and recall on the validation set.
