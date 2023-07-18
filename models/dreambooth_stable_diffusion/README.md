## models

This folder contains the Jupyter notebook `prompt_conditioned_dcgan.ipynb`, which is used to train the baseline model—a DCGAN (Deep Convolutional Generative Adversarial Network) conditioned on text prompts.

### Usage

To use the `prompt_conditioned_dcgan.ipynb` notebook, follow these steps:

1. Create a folder in your Colab workspace.
2. Define a session name for your training process.
3. Within the session folder, create two subfolders:
   - `instance_images`: This folder should contain the image instances for training the model. The images should be in the formats: .png, .jpeg, or .PNG.
   - `captions`: This folder should contain the corresponding text prompts for each image instance. Each text prompt should be in a separate .txt file.
     - The name of each .txt file in the `captions` folder should be unique, with the only difference being the file extension.
     - The number of text prompt files in the `captions` folder should match the number of image instances in the `instance_images` folder.
4. Open and run the `prompt_conditioned_dcgan.ipynb` notebook in Colab.
5. Follow the instructions in the notebook to train the model using the provided image instances and text prompts.

### License

The code provided in the `prompt_conditioned_dcgan.ipynb` notebook is released under the [MIT License](LICENSE).

### Acknowledgments

We would like to express our gratitude to the creators and contributors of various open-source libraries, models, and research that have influenced and supported the development of this project. In particular, we would like to acknowledge the following:

- The [DCGAN (Deep Convolutional Generative Adversarial Network)](https://arxiv.org/abs/1511.06434) model, which serves as the basis for the baseline model implemented in this project.
- The insightful tutorial and guidance provided by Laura Carnevali in her YouTube video on [Generative Adversarial Networks with PyTorch](https://www.youtube.com/watch?v=u6wxGrqIX5Y&ab_channel=LauraCarnevali). Her tutorial has been a valuable resource for understanding GANs and their implementation using PyTorch.
- The developers of the [PyTorch](https://pytorch.org/) library, which has been instrumental in training and running the deep learning models used in this project.
- The contributors to the [MusicCNN](https://github.com/jordipons/musicnn) project, which provides the music tagging functionality used to extract music tags from songs in this project.

We sincerely appreciate the efforts and contributions of these individuals and the wider open-source community in advancing the field of generative AI.

Please note that while this project has drawn inspiration and guidance from the mentioned resources, any errors or shortcomings are solely the responsibility of the project developers.
