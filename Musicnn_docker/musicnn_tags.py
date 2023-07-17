import sys


sys.path.append("C:/Users/jmartorr/venv/Lib/site-packages/")
from musicnn.tagger import top_tags


def extract(song):
    tops = top_tags(song, model='MTT_musicnn', topN=10)
    tags = ', '.join(tops)
    return tags


if __name__ == "__main__":
    song = '/data/song_infered.mp3'
    tops_string = extract(song)
    print(tops_string)
    text_file = open("/data/tags_infered.txt", "w")
    text_file.write(tops_string)
    text_file.close()
