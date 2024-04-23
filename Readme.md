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

Installing the required dependencies and modules

```
pip install -r requirements.txt
```

### How to train model

1. **Clone the repository:**

```
git clone https://github.com/harsh-tank/COMP6721-GroupR.git
```

2. **Install dependencies:**

```
pip install -r requirements.txt
```

3. **Launch Jupyter Notebook:**

```
jupyter notebook
```

4. **Open and run the notebooks:**
   - Navigate to the "notebooks" directory in the Jupyter Notebook interface.
   - Open a `.ipynb` file (notebook).
   - Run the notebook cells by clicking the "Run" button or using "Shift + Enter".

**Notes:**

- The dataset used might be available on Kaggle or Google Drive. Refer to the notebooks for download instructions (if necessary).
- Kaggle and Google Colab are convenient options as they eliminate the need to download datasets locally.

### Testing the Model

**Using a Pre-trained Model:**

1. **Open "ModelTesting.ipynb" from the "notebooks" folder.** This notebook contains the testing code.
2. **Provide the path to the trained model:** Edit the `torch.load()` function to specify the location of the model weights.
3. **Class label conversion (optional):** The `sat_map` dictionary (defined in the notebook) can be used to convert predicted outputs to class labels for dataset-1. Create similar dictionaries for other datasets, if applicable.
4. **Image classification:** Provide the path to an image using the `image.open()` method.
5. **Run the code:** The code will predict the class for the input image.
