from gensim.corpora import Dictionary
from smart_open import open

dictionary = Dictionary()

class Corpus:
	def __iter__(self):
		for line in open("assets/corpus.txt"):
			yield dictionary.doc2bow(line.lower().split())

