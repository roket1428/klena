# -*- coding: utf-8 -*-
import nltk
import unidecode
import smart_open

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

class Corpus:
	def __iter__(self):
		for line in open("corpus.txt"):
			yield sent_tokenize(line)

def normalize(corpus, limit):
	with open("corpus_fitered.txt", "w", encoding='utf-8') as out_f:
		for i,w in enumerate(corpus):
			if i == limit:
				break
			print("corpus:", i)
			words = word_tokenize(w[0])
			stop_words = set(stopwords.words('english'))

			filtered = [word for word in words if word not in stop_words]

			out = unidecode.unidecode(str(' '.join(filtered)))

			out_f.write(bytes(out, 'utf-8').decode('utf-8') + '\n')

def minimalize(corpus, limit):
	with open("corpus_fitered.txt", encoding='utf-8') as in_f:
		with open("corpus_minimal.txt", "w", encoding='utf-8') as out_f:
			for i, w in enumerate(in_f):
				if i == limit:
					break
				c_out = ""
				gene_pool = list("abcdefghijklmnopqrstuvwxyz[];\',./<\\")
				words = word_tokenize(w[:])
				for word in words:
					for c in word:
						if c not in gene_pool:
							continue
						c_out = f"{c_out}{c}"
					c_out = f"{c_out} "
				out_f.write(bytes(''.join(c_out), 'utf-8').decode('utf-8') + '\n')


corpus = Corpus()

normalize(corpus, 1000)
minimalize(corpus, 10)

