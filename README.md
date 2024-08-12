# Image Colorization Project

This project aims to colorize black and white images using deep learning techniques.

## Project Structure

- `data/`
  - `test_black/`: Contains black and white test images.
  - `test_color/`: Contains color test images.
  - `train_black/`: Contains black and white training images.
  - `train_color/`: Contains color training images.
- `best_colorization_model.h5`: The best model for colorization.
- `Image_colorization_model.h5`: The main model used for colorization.
- `unet.h5`: The U-Net model used in the project.
- `Data_Exploration.ipynb`: Notebook for exploring the dataset.
- `Image_colorization.ipynb`: Notebook for training and testing the colorization model.

## Installation

1. Clone the repository:
    ```sh
    git clone [<repository_url>](https://github.com/RishabhDE/image-colorization-via-transfer-learning.git)
    ```
2. Navigate to the project directory:
    ```sh
    cd image-colorization-via-transfer-learning
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Data Exploration

To explore the dataset, open and run the `Data_Exploration.ipynb` notebook. This notebook provides insights into the dataset, including visualizations and statistics.

### Image Colorization

To train and test the image colorization model, open and run the `Image_colorization.ipynb` notebook. This notebook includes the following steps:
1. Loading the dataset.
2. Preprocessing the images.
3. Defining and training the model.
4. Evaluating the model on test images.
5. Visualizing the results.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- The U-Net model architecture is inspired by the paper "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Olaf Ronneberger, Philipp Fischer, and Thomas Brox.
- Special thanks to the contributors and the open-source community for their valuable resources and support.
