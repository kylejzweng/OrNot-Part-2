import pandas as pd
from nltk import RegexpTokenizer
from nltk.text import ConcordanceIndex

df = pd.read_csv('full_ornot_dataset.csv')

# Dropping any remaining null headers
df.dropna(subset = ['headers'], inplace = True)
# print(df.info())

# Make headers and reviews lowercase
df['headers'] = df['headers'].str.lower()
df['body'] = df['body'].str.lower()

# Tokenization - splitting headers and reviews into lists of words. Also removing any punctuation
tokenizer = RegexpTokenizer("[\w]+")
df['tokenized_headers'] = df['headers'].map(tokenizer.tokenize)
df['tokenized_body'] = df['body'].map(tokenizer.tokenize)

flattened_headers = []
for sublist in df['tokenized_headers']:
    for word in sublist:
        flattened_headers.append(word)
    
flattened_reviews = []
for sublist in df['tokenized_body']:
    for word in sublist:
        flattened_reviews.append(word)

# # Finding instances where people say 'wish'
# concordance_index = ConcordanceIndex(flattened_reviews)

# iwish_concordance = concordance_index.find_concordance(['i', 'wish'])
# for concordance_line in iwish_concordance:
#     left_context = ' '.join(concordance_line.left)
#     matched_word = concordance_line.query
#     right_context = ' '.join(concordance_line.right)
#     print(f"Left Context: {left_context}")
#     print(f"Matched Word: {matched_word}")
#     print(f"Right Context: {right_context}")
#     print("============================================")
# print(len(iwish_concordance))   # 40

# ilovethe_concordance = concordance_index.find_concordance(['i', 'love', 'the'])
# for concordance_line in ilovethe_concordance:
#     left_context = ' '.join(concordance_line.left)
#     matched_word = concordance_line.query
#     right_context = ' '.join(concordance_line.right)
#     print(f"Left Context: {left_context}")
#     print(f"Matched Word: {matched_word}")
#     print(f"Right Context: {right_context}")
#     print("============================================")
# print(len(ilovethe_concordance))   # 163
# #                                 # They love the fabrice, color, fit, stretch/elasticity

# improve_concordance = concordance_index.find_concordance('improve')
# for concordance_line in improve_concordance:
#     left_context = ' '.join(concordance_line.left)
#     matched_word = concordance_line.query
#     right_context = ' '.join(concordance_line.right)
#     print(f"Left Context: {left_context}")
#     print(f"Matched Word: {matched_word}")
#     print(f"Right Context: {right_context}")
#     print("============================================")
# print(len(improve_concordance))   # 10

# onlything_concordance = concordance_index.find_concordance(['only', 'thing'])
# for concordance_line in onlything_concordance:
#     left_context = ' '.join(concordance_line.left)
#     matched_word = concordance_line.query
#     right_context = ' '.join(concordance_line.right)
#     print(f"Left Context: {left_context}")
#     print(f"Matched Word: {matched_word}")
#     print(f"Right Context: {right_context}")
#     print("============================================")
# print(len(onlything_concordance))   # 42

# price_concordance = concordance_index.find_concordance(['price'])
# for concordance_line in price_concordance:
#     left_context = ' '.join(concordance_line.left)
#     matched_word = concordance_line.query
#     right_context = ' '.join(concordance_line.right)
#     print(f"Left Context: {left_context}")
#     print(f"Matched Word: {matched_word}")
#     print(f"Right Context: {right_context}")
#     print("============================================")
# print(len(price_concordance))   # 134, price is high but seems worth it for most people

# buyagain_concordance = concordance_index.find_concordance(['buy', 'again'])
# for concordance_line in buyagain_concordance:
#     left_context = ' '.join(concordance_line.left)
#     matched_word = concordance_line.query
#     right_context = ' '.join(concordance_line.right)
#     print(f"Left Context: {left_context}")
#     print(f"Matched Word: {matched_word}")
#     print(f"Right Context: {right_context}")
#     print("============================================")
# print(len(buyagain_concordance))   # 38

# recommend_concordance = concordance_index.find_concordance(['recommend'])
# for concordance_line in recommend_concordance:
#     left_context = ' '.join(concordance_line.left)
#     matched_word = concordance_line.query
#     right_context = ' '.join(concordance_line.right)
#     print(f"Left Context: {left_context}")
#     print(f"Matched Word: {matched_word}")
#     print(f"Right Context: {right_context}")
#     print("============================================")
# print(len(recommend_concordance))   # 275

# customerservice_concordance = concordance_index.find_concordance(['customer', 'service'])
# for concordance_line in customerservice_concordance:
#     left_context = ' '.join(concordance_line.left)
#     matched_word = concordance_line.query
#     right_context = ' '.join(concordance_line.right)
#     print(f"Left Context: {left_context}")
#     print(f"Matched Word: {matched_word}")
#     print(f"Right Context: {right_context}")
#     print("============================================")
# print(len(customerservice_concordance))   # 50

# logos_concordance = concordance_index.find_concordance(['logos'])
# for concordance_line in logos_concordance:
#     left_context = ' '.join(concordance_line.left)
#     matched_word = concordance_line.query
#     right_context = ' '.join(concordance_line.right)
#     print(f"Left Context: {left_context}")
#     print(f"Matched Word: {matched_word}")
#     print(f"Right Context: {right_context}")
#     print("============================================")
# print(len(logos_concordance))   # 25

# madein_concordance = concordance_index.find_concordance(['made', 'in'])
# for concordance_line in madein_concordance:
#     left_context = ' '.join(concordance_line.left)
#     matched_word = concordance_line.query
#     right_context = ' '.join(concordance_line.right)
#     print(f"Left Context: {left_context}")
#     print(f"Matched Word: {matched_word}")
#     print(f"Right Context: {right_context}")
#     print("============================================")
# print(len(madein_concordance))   # 77

# donate_concordance = concordance_index.find_concordance(['donate'])
# for concordance_line in donate_concordance:
#     left_context = ' '.join(concordance_line.left)
#     matched_word = concordance_line.query
#     right_context = ' '.join(concordance_line.right)
#     print(f"Left Context: {left_context}")
#     print(f"Matched Word: {matched_word}")
#     print(f"Right Context: {right_context}")
#     print("============================================")
# print(len(donate_concordance))   # no hits for donate, charity, nonprofit

# zipper_concordance = concordance_index.find_concordance(['zipper'])
# for concordance_line in zipper_concordance:
#     left_context = ' '.join(concordance_line.left)
#     matched_word = concordance_line.query
#     right_context = ' '.join(concordance_line.right)
#     print(f"Left Context: {left_context}")
#     print(f"Matched Word: {matched_word}")
#     print(f"Right Context: {right_context}")
#     print("============================================")
# print(len(zipper_concordance))    #318


                                                   



