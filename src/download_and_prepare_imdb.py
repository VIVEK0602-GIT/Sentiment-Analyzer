import os
import pandas as pd
import requests
import zipfile

def download_imdb_dataset(dest_path='data/IMDB_Dataset.csv'):
    url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    local_tar = 'data/aclImdb_v1.tar.gz'
    extract_dir = 'data/aclImdb'
    if not os.path.exists(dest_path):
        print('Downloading IMDB dataset...')
        os.makedirs('data', exist_ok=True)
        r = requests.get(url, stream=True)
        with open(local_tar, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print('Extracting...')
        import tarfile
        with tarfile.open(local_tar, 'r:gz') as tar:
            tar.extractall(path='data')
        print('Processing...')
        # Read reviews from pos/neg folders
        def read_reviews(folder, sentiment):
            reviews = []
            folder_path = os.path.join(extract_dir, folder)
            for label in ['pos', 'neg']:
                label_path = os.path.join(folder_path, label)
                for fname in os.listdir(label_path):
                    with open(os.path.join(label_path, fname), encoding='utf-8') as f:
                        reviews.append({'review': f.read(), 'sentiment': label})
            return reviews
        train_reviews = read_reviews('train', 'train')
        test_reviews = read_reviews('test', 'test')
        all_reviews = train_reviews + test_reviews
        df = pd.DataFrame(all_reviews)
        df.to_csv(dest_path, index=False)
        print(f'Saved to {dest_path}')
    else:
        print(f'{dest_path} already exists.')

if __name__ == '__main__':
    download_imdb_dataset() 