import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/fasttext/labeled_queries.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
queries_df = pd.read_csv(queries_file_name)[['category', 'query']]
queries_df = queries_df[queries_df['category'].isin(categories)]
# queries_df = queries_df.head(1000)  # FOR TESTING ONLY

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
queries_df['query'] = queries_df['query'].str.lower()
queries_df['query'] = queries_df['query'].str.replace(r'[^a-zA-Z0-9]+', ' ')
queries_df['query'] = queries_df['query'].str.replace(r' +', ' ')
queries_df['query'] = queries_df['query'].apply(lambda x: ' '.join(stemmer.stem(w) for w in x.split()))
# print("ORIGINAL QUERIES DF")
# print(queries_df)
# print("GETTING VALUE COUNTS")
query_counts = queries_df['category'].value_counts()
# print("QUERY COUNTS")
# print(query_counts)

# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.
# print("GETTING CATEGORIES BELOW THRESHOLD")
categories_below_threshold = query_counts[query_counts < min_queries]
# print("CATEGORIES BELOW QUERIES")
# print(categories_below_threshold)
# print(categories_below_threshold.index)
# print("PARENTS DF")
# print(parents_df)
print(f'{len(categories_below_threshold)} categories with query counts below threshold')

while len(categories_below_threshold) > 0:
    # get parents of categories below threshold
    parents = parents_df[parents_df['category'].isin(categories_below_threshold.index)].set_index('category')
    # print("PARENTS")
    # print(parents)
    # replace categories in queries_df with parents
    # print("QUERIES")
    # print(queries_df)
    def parent_lookup(cat):
        try:
            parent_value = parents.loc[cat][0]
            # print(f"FOUND PARENT OF {cat}: {parent_value}")
        except KeyError:
            return cat
    queries_df['category'] = queries_df['category'].apply(parent_lookup)
    # queries_df = queries_df.join(parents, on='category', how='left', rsuffix='rollup_cat')
    # print("JOINED QUERY DF")
    # print(queries_df)

    # get queries below thresholds again
    # print("GETTING VALUE COUNTS")
    query_counts = queries_df['category'].value_counts()
    # print("GETTING CATEGORIES BELOW THRESHOLD")
    categories_below_threshold = query_counts[query_counts < min_queries]
    print(f'{len(categories_below_threshold)} categories with query counts below threshold')

# print("FINAL QUERIES DF")
# print(queries_df)
# print("QUERY COUNTS")
print("Final unique categories by query count")
print(queries_df['category'].value_counts())
# Create labels in fastText format.
queries_df['label'] = '__label__' + queries_df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
queries_df = queries_df[queries_df['category'].isin(categories)]
queries_df['output'] = queries_df['label'] + ' ' + queries_df['query']
queries_df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
