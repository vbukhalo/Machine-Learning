import pandas as pd
import re
from nltk.stem.porter import PorterStemmer 


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

def tokenizer(text):
	return text.split()

porter = PorterStemmer()
def tokenizer_porter(text):
	return [porter.stem(word) for word in text.split()]

print(tokenizer_porter('runners like running and thus they run'))


df = pd.read_csv('movie_data.csv', encoding='utf-8')

df['review'] = df['review'].apply(preprocessor)

