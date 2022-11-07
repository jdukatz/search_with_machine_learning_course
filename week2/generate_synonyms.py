import argparse
import fasttext

parser = argparse.ArgumentParser()
general = parser.add_argument_group('general')
general.add_argument('--cutoff', default=0.75, type=float)

args = parser.parse_args()
similarity_cutoff = args.cutoff

model = fasttext.load_model('/workspace/datasets/fasttext/normalized_title_model.bin')

synonyms_list = []
top_words_file = '/workspace/datasets/fasttext/top_words.txt'
output_file = '/workspace/datasets/fasttext/synonyms.csv'
with open(top_words_file) as f:
    for top_word in f:
        synonyms = model.get_nearest_neighbors(top_word.strip())
        synonyms_out = ','.join([top_word.strip()] + [s[1] for s in synonyms if s[0] >= similarity_cutoff])
        synonyms_list.append(synonyms_out)

with open(output_file, 'w') as f:
    for line in synonyms_list:
        f.write(line + '\n')
