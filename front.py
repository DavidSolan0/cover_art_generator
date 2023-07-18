from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask import abort, request
import docker
import requests
import argparse
from PIL import Image
from io import BytesIO
import os

# parser
parser = argparse.ArgumentParser(description='Descripción de tu script')
parser.add_argument('url', type=str, help='URL a procesar')
args = parser.parse_args()
nurl = args.url

app = Flask(__name__)
bootstrap = Bootstrap(app)


@app.before_request
def limit_remote_addr():
    if request.remote_addr != '37.223.179.210':
        abort(403)  # Forbidden


@app.route('/cover', methods=['GET', 'POST'])
def index():
    prompt = 'prompt'
    return render_template('index.html', bootstrap=bootstrap, value=prompt)


@app.route('/new', methods=['POST'])
def new():
    return render_template('index.html', bootstrap=bootstrap)


@app.route('/')
def wellcome():
    return "Cover Art Generation (UPC)"


def run_cnn():
    """
    Execute process Musicnn, this part of the code must be modified to avoid lift a new container in every process
    :return:  Top Tags
    """
    print('Docker Processing...')
    client = docker.from_env()
    top_string = ''
    try:
        container = client.containers.run(
            'musicnn',
            name='mcnn',
            volumes={
                f"/var/www/cover/cover-art-generation/inference_songs": {
                  'bind': '/data',
                  'mode': 'rw'
                  }
            },
            detach=True
        )
        container.wait()
        with open('/var/www/cover/cover-art-generation/inference_songs/tags_infered.txt', 'r') as tags:
            top_string = tags.read()

        print('Procés completat al contenidor: {}'.format(container.id))
        container.stop()
        container.remove()

    except Exception as e:
        print(e)

    return top_string


def extract_prompt(style, artist, title, year, top_tags):
    """
    Adapts de parameters to the prompt
    :param style: tag from the creator
    :param artist: name of the artist
    :param title: name of the album/song
    :param year: year of release
    :param top_tags: tags from musicnn
    :return: prompt
    """
    prompt = (
        f"cover album of {style} song, titled {title}, "
        f"released in {year}, by {artist}, "
        f"with the following music tags {top_tags}"
    )
    return prompt


@app.route('/upload', methods=['POST'])
def upload():
    """
    Get the values from the html inputs and transforms in the prompt
    :return: prompt
    """
    artist = request.form.get('artist')
    year = request.form.get('year')
    title = request.form.get('title')
    style = request.form.get('style')
    print('First prompt {}'.format(request.form.get('prompt')))

    # get song from requests
    song = request.files['song']

    # save song in inference path
    print('Saving song....')
    song.save('/var/www/cover/cover-art-generation/inference_songs/song_infered.mp3')

    # get top tags from Musicnn Docker
    top_tags = run_cnn()
    prompt = extract_prompt(style, artist, title, year, top_tags)
    return render_template('index.html', value=prompt)


def get_images(imagen_stream):
    """
    Pass bytes to image
    :param imagen_stream:
    :return: Image
    """
    imagen_stream = BytesIO(imagen_stream.content)
    images_recuperadas = []
    while True:
        try:
            print('Processing images...')
            # Leer los bytes de la imagen
            bytes_imagen = bytearray(imagen_stream.read())
            if not bytes_imagen:
                break
            imagen = Image.open(BytesIO(bytes_imagen))
            images_recuperadas.append(imagen)

        except Exception as e:
            print(f"Error al procesar una imagen: {e}")
            
    print(len(images_recuperadas))
    return images_recuperadas


@app.route('/cover_album', methods=['POST'])
def cover_album():
    """
    Function to process the Difussion model and return the cover Album
    :return: html with cover
    """
    prompt = request.form.get('prompt')
    print('prompt generate {}'.format(prompt))

    url = nurl

    images_bytes = requests.post(url, data=prompt)

    images_ls = get_images(images_bytes)

    for iterator, im in enumerate(images_ls):
        im.save(f'/var/www/cover/cover-art-generation/static/images/cover{iterator}.jpeg')

    return render_template('cover_generated.html', bootstrap=bootstrap)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8080')
