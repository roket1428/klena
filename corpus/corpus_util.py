#	project-e find the most efficient keyboard layout using the genetic algorithm
#		Copyright (C) 2023 roket1428 <meorhan@protonmail.com>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import nltk
import unidecode
import smart_open

from gensim.corpora import WikiCorpus
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

# iterator class for the big corpus file
class Corpus:
	def __iter__(self):
		for line in open("corpus.txt"):
			yield sent_tokenize(line)

# extract the corpus from wikipedia articles dump
def extract_corpus():
	wiki = WikiCorpus("enwiki-latest-pages-articles1.xml-p1p41242.bz2")
	with open("corpus.txt", "w", encoding='utf-8') as out_f:
		for text in wiki.get_texts():
			out_f.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')

# reduce corpus dimensions by filtering out stop words and stemming
def reduce_features(corpus):
	with open("corpus_reduced.txt", "w", encoding='utf-8') as out_f:
		for i, w in enumerate(corpus):
			if i == 50:
				break
			print("i:", i)
			words = word_tokenize(w[0])
			stop_words = set(stopwords.words("english"))
			filtered_stop_words = [word for word in words if word not in stop_words]
			decoded = unidecode.unidecode(str(' '.join(filtered_stop_words)))
			stemmed_words = [PorterStemmer().stem(word) for word in decoded.split(' ')]

			out_f.write(bytes(' '.join(stemmed_words), 'utf-8').decode('utf-8') + '\n')

# remove the duplicate words to further the reduction
def remove_duplicates():
	duplicates = []
	cleaned = []
	with open("corpus_reduced.txt", "r") as in_f:
		with open("corpus_cleaned.txt", "w", encoding='utf-8') as out_f:
			for s in in_f:
				for w in s.split(' '):
					if w in cleaned:
						if w in duplicates:
							continue
						else:
							duplicates.append(w)
					else:
						cleaned.append(w)
			out_f.write(bytes(''.join(cleaned), 'utf-8').decode('utf-8'))

# lastly, filter out unwanted characters
def filter_out():
	with open("corpus_cleaned.txt", "r") as in_f:
		with open("corpus_filtered.txt", "w", encoding='utf-8') as out_f:
			input = in_f.read()
			input.replace('\n', '').replace(' ', '')

			gene_pool = list("abcdefghijklmnopqrstuvwxyz[];\',./<\\")

			for c in input:
				if c not in gene_pool:
					continue
				else:
					out_f.write(bytes(c, 'utf-8').decode('utf-8'))


corpus = Corpus()

reduce_features(corpus)
remove_duplicates()
filter_out()
