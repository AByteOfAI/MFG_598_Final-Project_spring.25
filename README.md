# MFG_598_Final-Project_spring.25
Hi I am Abhijit Sinha and this is my Final Project .
# Fake News Detection using LSTM

This project implements a deep learning model using Bidirectional Long Short-Term Memory (Bi-LSTM) networks to classify news articles as either "Real" or "Fake".

## Table of Contents

* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Dependencies](#dependencies)
* [Setup](#setup)
* [Usage](#usage)
* [Model Architecture](#model-architecture)
* [Evaluation](#evaluation)
* [Generated Artifacts](#generated-artifacts)
* [References](#references)
* [Author](#author)
* [License](#license)

## Project Overview

The goal of this project is to build and train a robust classifier capable of distinguishing between legitimate news articles and fabricated ones. It utilizes natural language processing (NLP) techniques for text preprocessing and a Bi-LSTM model implemented in TensorFlow/Keras for classification.

## Dataset

The model is trained on a dataset composed of two CSV files:

1.  `Fake.csv`: Contains news articles labeled as fake.
2.  `True.csv`: Contains news articles labeled as real.

These files are expected to contain columns like `title`, `text`, `subject`, and `date`. The script combines these datasets, adds a `target` column (0 for Fake, 1 for Real), shuffles the data, and performs preprocessing on the `text` column.

* **Preprocessing Steps:**
    * Convert text to lowercase.
    * Remove punctuation.
    * Remove common English stopwords (using NLTK).

*Note: You will need to update the paths to `Fake.csv` and `True.csv` within the `FINAL_PROJECT_FAKENEWSDETECTION.ipynb` notebook (`DATA_FAKE_PATH` and `DATA_TRUE_PATH` variables) to match their location on your system.*

## Dependencies

The project relies on the following Python libraries:

* Python 3.x
* NumPy
* Pandas
* NLTK (Natural Language Toolkit)
* Scikit-learn
* TensorFlow (>=2.x)
* Bokeh (for plotting)
* WordCloud
* Matplotlib

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AByteOfAI/MFG_598_Final-Project_spring.25
    cd https://github.com/AByteOfAI/MFG_598_Final-Project_spring.25
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file with the following content:
    ```txt
    numpy
    pandas
    nltk
    scikit-learn
    tensorflow
    bokeh
    wordcloud
    matplotlib
    jupyter # If you want to run the notebook
    ```
    Then install using pip:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK Stopwords:**
    Run the following Python code once to download the necessary NLTK data:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

## Usage

1.  **Update Data Paths:** Open the `FINAL_PROJECT_FAKENEWSDETECTION.ipynb` notebook and modify the `DATA_FAKE_PATH` and `DATA_TRUE_PATH` variables to point to your `Fake.csv` and `True.csv` files.

2.  **Run the Notebook:** Execute the cells in the `FINAL_PROJECT_FAKENEWSDETECTION.ipynb` notebook sequentially. This will:
    * Load and preprocess the data.
    * Build the tokenizer.
    * Define and compile the Bi-LSTM model.
    * Train the model using the training data and validate it on the validation set.
    * Evaluate the trained model on the test set.
    * Save the trained model, tokenizer, and generate visualizations.

3.  **View Artifacts:** All generated outputs (model, tokenizer, plots, logs) will be saved in the `artifacts/` directory.

## Model Architecture

The classification model is a Bidirectional LSTM network built with TensorFlow/Keras:

1.  **Input Layer:** Takes sequences of token IDs (padded to `MAX_SEQUENCE_LENGTH = 300`).
2.  **Embedding Layer:** Maps token IDs to dense vectors (`VOCAB_SIZE = 40000`, `EMBED_DIM = 128`). Uses `mask_zero=True`.
3.  **Dropout:** Applies dropout (`DROPOUT = 0.30`) to the embeddings.
4.  **Bidirectional LSTM:** Processes sequences in both forward and backward directions (`LSTM_UNITS = 128`).
5.  **Layer Normalization:** Normalizes the activations from the LSTM layer.
6.  **Dropout:** Applies another dropout layer (`DROPOUT = 0.30`).
7.  **Output Layer:** A single Dense neuron with a sigmoid activation function for binary classification (Fake/Real).

The model is compiled using the Adam optimizer (`LEARNING_RATE = 2e-4`) and `binary_crossentropy` loss.

## Evaluation

The model's performance is evaluated on a held-out test set using the following metrics:

* **Loss:** Binary Cross-Entropy
* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**

The final test results achieved during the notebook run were approximately:
* **Test Loss:** 0.0167
* **Test Accuracy:** 0.9963
* **Test Precision:** 0.9972
* **Test Recall:** 0.9950
* **Test F1-Score:** 0.9961

Visualizations including a training dashboard (Loss, Accuracy, F1 vs. Epochs) and a confusion matrix are generated to provide further insight into the model's performance.

## Generated Artifacts

Upon successful execution, the following files are saved in the `artifacts/` directory:

* `fakenews_lstm.keras`: The trained Keras model file.
* `tokenizer.json`: The fitted Keras Tokenizer configuration.
* `plots/`:
    * `training_dashboard.html`: An interactive Bokeh plot showing training/validation loss, accuracy, and F1-score over epochs.
    * `confusion_matrix.html`: An interactive Bokeh plot visualizing the confusion matrix on the test set.
    * `wc_fake.png`: A word cloud generated from the text of fake news articles.
    * `wc_real.png`: A word cloud generated from the text of real news articles.
* `logs/`: Contains TensorBoard logs for monitoring training progress (can be viewed using `tensorboard --logdir artifacts/logs`).

## References

* **Dataset Source:** [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
* **TensorFlow:** Abadi, M., et al. (2015). TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems. *arXiv preprint arXiv:1603.04467*.
* **Keras:** Chollet, F., et al. (2015). Keras. *GitHub repository*. https://github.com/keras-team/keras
* **NLTK:** Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media Inc.
* **Scikit-learn:** Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, *12*, 2825-2830.
* **Pandas:** McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, *445*, 51-56.
* **Bokeh:** Bokeh Development Team (2023). Bokeh: Python library for interactive visualization. https://bokeh.org
* **WordCloud:** Mueller, A. (2018). word_cloud. *GitHub repository*. https://github.com/amueller/word_cloud

## Author
- Abhijit Sinha
- Arizona State University
- email: asinh117@asu.edu
- Youtube video with demonstration: https://youtu.be/x1Rbk_dKNxw?si=g1PSldtYjI3HcwQ2

## License

This project is licensed under the MIT License. You can find the full license text in the 'LICENSE.md' file.
