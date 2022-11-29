#!/bin/sh

# remove that annoying '
cat corpus_minimal.txt | sed "s/'//g" > corpus_minimal.txt.tmp

rm corpus_minimal.txt

mv corpus_minimal.txt.tmp corpus_minimal.txt
