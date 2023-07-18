# Generative AI for Album Cover Creation

This GitHub repository contains code and resources for generating album covers using generative AI techniques. The project aims to extract music tags from songs using MusicCNN, and then utilizes the album name, singer name, and release date to create a prompt for conditioning the generative model.

## Repository Structure

The repository is organized into the following folders:

### musicnn_docker
- **dockerfile**: Dockerfile used to create the Docker image for running the MusicCNN tool.
- **musicnn_tags.py**: Utility Python module to extract music tags from songs using MusicCNN.

### data_collection
- **api_extract_jamendo.ipynb**: Jupyter notebook to extract training data from the Jamendo API.
- **api_extract_spotify.ipynb**: Jupyter notebook to extract training data from the Spotify API.

### data_preprocessing
- **extract_musicnn.py**: Python module containing code to extract text prompts from music tags for each song.

### models
- **prompt_conditioned_dcgan.ipynb**: Jupyter notebook for training the baseline model, a DCGAN conditioned on text prompts.
- **fast_dreambooth.ipynb**: Jupyter notebook for training the production model, which involves finetuning Stable Diffusion using Lora.

## Usage

To generate album covers using this repository, follow these steps:

1. Set up the MusicCNN Docker container by building the Docker image using the provided Dockerfile in the `musicnn_docker` folder.
2. Extract the necessary training data by running the appropriate Jupyter notebooks in the `data_collection` folder. Use `api_extract_jamendo.ipynb` for extracting data from the Jamendo API and `api_extract_spotify.ipynb` for extracting data from the Spotify API.
3. Preprocess the extracted data using the `extract_musicnn.py` module in the `data_preprocessing` folder. This module will extract text prompts from the music tags for each song.
4. Train the generative models using the provided Jupyter notebooks in the `models` folder. The `prompt_conditioned_dcgan.ipynb` notebook trains the baseline model, which is a DCGAN conditioned on text prompts. The `fast_dreambooth.ipynb` notebook trains the production model by finetuning Stable Diffusion using Lora.
5. Once the models are trained, deploy the web page provided to generate album covers. The web page should allow users to upload a song, enter the metadata (singer name, release date, and song name), and obtain the corresponding album cover.

## Requirements

To run this project, you need the following dependencies:

- Docker: to build and run the MusicCNN Docker container.
- Python 3.x: to execute the provided Python modules and Jupyter notebooks.
- PyTorch: for training and running the generative models.

Please refer to the documentation and instructions within each notebook for more detailed usage and setup information.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

The development of this project was influenced by various open-source libraries, models, and research. We would like to acknowledge their contributions to the field of generative AI.
