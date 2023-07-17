import pandas as pd
from musicnn.tagger import top_tags

def extract(path):
    """
    Enter the dataset with the audios and extract all the tags
    :param path: dataset_path
    :return: dataset with tags
    """
    dataset_spotify = pd.read_csv(path)
    dataset_spotify = dataset_spotify.iloc[:,1:]
    dataset_spotify['music tags'] = None
    for iterator, row in dataset_spotify.iterrows():
        print('iterator {}'.format(iterator))
        path_music = "C:/Users/jmartorr/desktop/deepl/Cover Project/audio_tracks/"
        tops = top_tags(path_music + row['Audio ID'] + '.mp3', model='MTT_musicnn', topN=10)
        tops_string = ', '.join(tops)
        dataset_spotify.at[iterator, 'music tags'] = tops_string
    return dataset_spotify


if __name__ == "__main__":
    dataset = extract('C:/Users/jmartorr/desktop/deepl/Cover Project/spotify_dataset.csv')
    dataset.to_csv('C:/Users/jmartorr/desktop/deepl/Cover Project/spotify_dataset_cnn.csv', index=False)
    dataset.to_pickle('C:/Users/jmartorr/desktop/deepl/Cover Project/spotify_dataset_cnn.pkl')