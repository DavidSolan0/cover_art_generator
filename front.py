from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask import abort, request
import docker
import requests
import argparse
from PIL import Image
from io import BytesIO
import os

# Crear el analizador de argumentos
parser = argparse.ArgumentParser(description='Descripción de tu script')

# Definir el argumento que deseas recoger
parser.add_argument('url', type=str, help='URL a procesar')

# Analizar los argumentos de la línea de comandos
args = parser.parse_args()

# Acceder al valor del parámetro URL
nurl = args.url

app = Flask(__name__)
bootstrap = Bootstrap(app)

# app.template_folder = '/var/www/cover/covert-art-generation/'

"""
@app.before_request
def limit_remote_addr():
    if request.remote_addr != '31.4.135.190':
        abort(403)  # Forbidden
"""

@app.route('/cover', methods=['GET', 'POST'])
def index():
    prompt = 'prompt'
    return render_template('index.html', bootstrap=bootstrap, value=prompt)

@app.route('/new', methods=['POST'])
def new():
    return render_template('index.html', bootstrap=bootstrap)

@app.route('/')
def hello():
    return "Cover Art Generation (UPC)"

def run_cnn(song_name=''):
    """
    Executa el proces de MusiCNN
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
        # Esperamos a que el contenedor termine
        container.wait()
        # Recupera el log de salida del contenedor
        with open('/var/www/cover/cover-art-generation/inference_songs/tags_infered.txt', 'r') as tags:
            # Lee el contenido del archivo
            top_string = tags.read()
        print('Procés completat al contenidor: {}'.format(container.id))
        container.stop()
        container.remove()
    except Exception as e:
        print(e)
    return top_string

def extract_prompt(style, artist, title, year, top_tags):
    prompt = (
        f"cover album of {style} song, titled {title}, "
        f"released in {year}, by {artist}, "
        f"with the following music tags {top_tags}"
    )
    return prompt


@app.route('/upload', methods=['POST'])
def upload():
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

    #return render_template('loaded.html', bootstrap=bootstrap)

def get_images(imagen_stream):
    imagen_stream = BytesIO(imagen_stream.content)
    images_recuperadas = []
    while True:
        try:
            print('Processing images...')
            # Leer los bytes de la imagen
            bytes_imagen = bytearray(imagen_stream.read())
            # Si no hay más bytes, salir del bucle
            if not bytes_imagen:
                break
            # Convertir los bytes en una imagen de PIL
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
    :return:
    """
    prompt = request.form.get('prompt')
    print('prompt generate {}'.format(prompt))

    #url = "http://ef9e-35-185-97-56.ngrok-free.app"
    url = nurl

    params = {
        'prompt':prompt
    }

    images_bytes = requests.post(url, data=prompt)

    images_ls = get_images(images_bytes)

    for iterator, im in enumerate(images_ls):
        image_path = os.path.join('static', 'images', f'cover{iterator}.jpeg')
        im.save(f'/var/www/cover/cover-art-generation/static/images/cover{iterator}.jpeg')

    return render_template('cover_generated.html', bootstrap=bootstrap)


if __name__ == '__main__':
    import getpass
    username = getpass.getuser()
    print("El usuario actual es:", username)
    app.run(host='0.0.0.0', port='8080')

    """
    imagen = Image.open(BytesIO(images_bytes.content))
    image_path = os.path.join('static', 'images', 'cover0.jpeg')
    imagen.save('/var/www/cover/cover-art-generation/static/images/cover0.jpeg')
    """
