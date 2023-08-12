import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re

pg1_df = pd.read_csv('ornotdata_pg1.csv')
pg2_df = pd.read_csv('ornotdata_pg2.csv')
pg3_df = pd.read_csv('ornotdata_pg3.csv')
pg4_df = pd.read_csv('ornotdata_pg4.csv')
pg5_df = pd.read_csv('ornotdata_pg5.csv')
trouble_df = pd.read_csv('ornotdata_trouble1.csv')

pd.set_option('display.max_columns', None)

# print(pg1_df.info())
# print(pg2_df.info())
# print(pg3_df.info())
# print(pg4_df.info())
# print(pg5_df.info())
# print(trouble_df.info())

# Combining all datasets into one
ornot_df = pd.concat([pg1_df, pg2_df, pg3_df, pg4_df, pg5_df, trouble_df], ignore_index = True)

# Dropping the first column. This was just an index when exporting the original data to a .csv file.
ornot_df = ornot_df[ornot_df.columns[1:]]
# print(ornot_df.info())

# Dropping any rows with all null data
ornot_df.dropna(how = 'all', inplace = True)
# print(ornot_df.info())

# Dropping any duplicate entries
ornot_df.drop_duplicates()
# print(ornot_df.info())

# I still have two rows with a null date.
null_date = ornot_df[ornot_df['date'].isnull()]
# print(null_date)

# All the other data is null too. I'm going to drop these rows
ornot_df.dropna(subset = ['date'], inplace = True)
# print(ornot_df.info())

# We now have a username and date for every row, but we're still missing data from important attributes
# like star, headers, body, and product. I'm going to check these out one by one.
null_star = ornot_df[ornot_df['star'].isnull()]
# print(null_star)

# These only have username, date, and location. Dropping these rows
ornot_df.dropna(subset = ['star'], inplace = True)
# print(ornot_df.info())

# Checking the null headers
null_header = ornot_df[ornot_df['headers'].isnull()]
# print(null_header)

# Okay, these are fine. I didn't know you could leave a review without a header, but you can. Keeping.
# Checking any null reviews
null_review = ornot_df[ornot_df['body'].isnull()]
# print(null_review)

# These aren't worth keeping. Dropping
ornot_df.dropna(subset = ['body'], inplace = True)
# print(ornot_df.info())

# Lastly, checking the product type that was reviewed
null_product = ornot_df[ornot_df['product'].isnull()]
# print(null_product)

# It'll be hard to run analysis if I don't know the product getting reviewed. Dropping
ornot_df.dropna(subset = ['product'], inplace = True)
# print(ornot_df.info())

# We now have all non-null data in the users, date, star, body, and product columns.

# Changing date datatype and dropping any that can't be converted
ornot_df['date'] = pd.to_datetime(ornot_df['date'], errors = 'coerce')
ornot_df.dropna(subset = ['date'], inplace = True)
# print(ornot_df.info())

# Changing fit column to float
ornot_df['fit'] = pd.to_numeric(ornot_df['fit'], errors = 'coerce')
print(ornot_df.info())      # Working with 5,285 reviews

# The height and weight column will need to be cleaned eventually, but that's going to be really tough. 
# Skipping for now

# Average star ratings for all reviews
average_star_rating = ornot_df['star'].mean()
# print(average_star_rating)  # Average star rating: 4.86

# Plotting all star ratings
star_count = ornot_df['star'].value_counts()
star_count.plot(kind = 'bar', figsize = (12, 8), color = 'royalblue')
plt.title('Star Rating Distribution', fontweight = 'bold')
plt.xlabel('Star Rating')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=0)
plt.tight_layout
plt.show()

# Average fit of all products
average_fit = ornot_df['fit'].mean()
# print(average_fit)  # Average fit : 2.90        3 = perfect fit, so just a little smaller 

# Fit distribution
fit_distribution = ornot_df['fit'].value_counts()
order = [1, 2, 3, 4, 5]
fit_distribution = fit_distribution.reindex(order)

ax = fit_distribution.plot(kind = 'bar', figsize = (12, 8), color = 'royalblue')
ax.set_xlabel('Fit Rating')     
ax.set_ylabel('Number of Reviews')         
ax.set_title('Fit Rating Distribution', fontweight = 'bold')  
new_x_labels = ['small', 'small-ish', 'perfect', 'big-ish', 'big']
ax.set_xticklabels(new_x_labels, rotation = 0)
plt.tight_layout     
plt.show()       

# Where are reviewers from?
location_count = ornot_df['location'].value_counts()
# print(location_count)
# print('Total Locations: ', location_count.sum())    # 94.98% of all buyers are from US. Canada, Australia, and UK round of top 4

# How many products reviewed by year
ornot_df['year'] = ornot_df['date'].dt.year
product_count_by_year = ornot_df['year'].value_counts().sort_index()
product_count_by_year.plot(kind = 'bar', figsize = (12, 6), color = 'royalblue')
plt.title('Number of Products Reviewed by Year', fontweight = 'bold')
plt.xlabel('Year')
plt.ylabel('Number of Products Reviewed')
plt.xticks(rotation=0)
plt.xticks(fontsize=10)
plt.tight_layout()
plt.show()

# How many reviewed of each prodcut
ornot_df['product'] = ornot_df['product'].str.lower()
product_count = ornot_df['product'].value_counts()
print(product_count)

# Grouping products by broader category i.e jersey, bib, sock...
grouping_products = {
    'Bibs/Tights': [r'bib', r'leg warmer'],
    'Jerseys': [r'jersey', r'base layer'],
    'Jackets/Vests': [r'jacket', r'vest'],
    'Shirts/Pullovers': [r'shirt', r'pullover'],
    'Shorts/Pants': [r'mission', r'boxer'],
    'Socks/Caps/Hat/Gloves': [r'sock', r'cap', r'hat', r'shoe', r'glove', r'beanie', r'neck'],
#   'Other': [r'dynaplug', r'gift', r'cygolite', r'belt', r'tool', r'topeak', r'bag', r'kom']
}

def map_product_category(product):
    product = product.lower()
    for category, key_words in grouping_products.items():
        for key_word in key_words:
            if re.search(key_word, product):
                return category
    return 'Other'

ornot_df['product_category'] = ornot_df['product'].map(map_product_category)
product_count_by_category = ornot_df['product_category'].value_counts()
product_count_by_category.plot(kind = 'bar', figsize = (12, 8), color = 'royalblue')
plt.xlabel('Product Category')
plt.ylabel('Number of Products')
plt.title('Number of Products Reviewed by Category', fontweight = 'bold')
plt.xticks(fontsize=10)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Average star rating per product category
average_star_by_category = ornot_df.groupby('product_category')['star'].mean()
print(average_star_by_category)     # They're basically all rated the same

# What sizes are most commonly reviewed
# Grouping sizes
grouping_sizes = {
    'x small': [r'extra small', r'xs'],
    'small': [r'small', r'sm', r'mens small'],
    'medium': [r'medium', r'md', r'medium / synthetic', r'mens medium', r'medium / merino wool'],
    'xx large': [r'xx-large', r'xx - large', r'xx large', r'xxl', r'xx- large'],
    'x large': [r'extra large', r'x-large', r'xl', r'mens x-large'],
    'large': [r'large', r'lg', r'mens large']
}

def map_size_category(size):
    if pd.notna(size):
        size = size.lower()
        for category, key_words in grouping_sizes.items():
            for key_word in key_words:
                if re.search(key_word, size):
                    return category
        return size
    else:
        return 'No size provided by reviewer'

ornot_df['size_category'] = ornot_df['size'].map(map_size_category)

pd.set_option('display.max_rows', None)

size_count = ornot_df['size_category'].value_counts()

# Most common sizes for shirts, shorts, and pants
cycling_clothes_sizes = ['x small', 'small', 'medium', 'large', 'x large', 'xx large']
short_sizes = ['28', '30', '32', '33', '34', '36', '38']
pant_sizes = ['28x32', '30x32', '30x34', '32x32', '32x34', '34x32', '34x34', '36x34', '38x34']
cycling_clothes_size_count = size_count[size_count.index.isin(cycling_clothes_sizes)]
short_size_count = size_count[size_count.index.isin(short_sizes)]
pant_size_count = size_count[size_count.index.isin(pant_sizes)]

# print('Bike Specific Clothing (jerseys, bibs, ...): ', cycling_clothes_size_count)
# print('Short Sizes: ', short_size_count)
# print('Pant Sizes: ', pant_size_count)

# Plotting all sizes
plt.figure(figsize=(15, 5))

# Reindexing sizes so they graph better
cycling_clothes_size_count_sorted = cycling_clothes_size_count.reindex(cycling_clothes_sizes)
short_size_count_sorted = short_size_count.reindex(short_sizes)
pant_size_count_sorted = pant_size_count.reindex(pant_sizes)

# Cycling clothes sizes
plt.subplot(1, 3, 1)
cycling_clothes_size_count_sorted.plot(kind = 'bar', color = 'royalblue')
plt.title('Jerseys, Bibs, etc.')
plt.xlabel('Size')
plt.ylabel('Count')
plt.xticks(rotation=45)

# Short sizes
plt.subplot(1, 3, 2)
short_size_count_sorted.plot(kind = 'bar', color = 'seagreen')
plt.title('Shorts')
plt.xlabel('Size')
plt.xticks(rotation=45)

# Pant sizes
plt.subplot(1, 3, 3)
pant_size_count_sorted.plot(kind = 'bar', color = 'lightcoral')
plt.title('Pants')
plt.xlabel('Size')
plt.xticks(rotation=45)
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.suptitle('Sizing Distributions by Category', fontweight = 'bold')
plt.tight_layout()
plt.show()

ornot_df.to_csv('full_ornot_dataset.csv', encoding = 'utf-8-sig')