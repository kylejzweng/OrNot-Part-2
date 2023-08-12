from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

driver = webdriver.Chrome()

# Function to scrape reviews from a given product URL
def scrape_reviews(product_url):

    driver.get(product_url)
    time.sleep(5)
        
    try:
        find_popup = driver.find_element(By.CSS_SELECTOR, 'button.needsclick.klaviyo-close-form.kl-private-reset-css-Xuajs1')
        find_popup.click()
        time.sleep(3)
    except Exception as e:
        print("Error while closing popup:", e)

    # Initialize lists to store the extracted information
    users = []
    review_post_date = []
    location_of_review = []
    star_ratings = []
    review_headers = []
    review_body = []
    fit_rating = []
    products = []
    size_ordered = []
    reviewers_height_and_weight = []

    while True:

        # Find all elements with the class 'stamped-review'
        review_elements = driver.find_elements(By.CLASS_NAME, 'stamped-review')

        # Loop through each 'stamped-review' element and extract the information
        for review in review_elements:

            # Extract the username from the 'strong' tag with the class 'author'
            try:
                username_element = review.find_element(By.CLASS_NAME, 'author')
                username = username_element.text
                users.append(username)
            except:
                users.append('N/A')

            # Extracting date of review when posted if available
            try:
                date_element = review.find_element(By.CLASS_NAME, 'created')
                date = date_element.text
                review_post_date.append(date)
            except:
                review_post_date.append('N/A')

            # Extracting location of review if available
            try:
                location_element = review.find_element(By.CSS_SELECTOR, '.review-location')
                location = location_element.text
                location_of_review.append(location)
            except:
                location_of_review.append('N/A')

            # Extracting star rating if available
            try:
                star_rating_element = review.find_element(By.CSS_SELECTOR, '.stamped-starratings')
                star= star_rating_element.get_attribute('data-rating')
                star_ratings.append(star)
            except:
                star_ratings.append('N/A')

            # Extracting heading of review if available
            try:
                header_element = review.find_element(By.CLASS_NAME, 'stamped-review-header-title')
                header = header_element.text
                review_headers.append(header)
            except:
                review_headers.append('N/A')

            # Extracting body of review if available
            try:
                review_body_element = review.find_element(By.CLASS_NAME, 'stamped-review-content-body')
                body = review_body_element.text
                review_body.append(body)
            except:
                review_body.append('N/A')
                
            # Extracting fit of product if available
            try:
                size_element = review.find_element(By.CSS_SELECTOR, 'div.stamped-review-variant')
                size = size_element.text
                size_ordered.append(size)
            except:
                size_ordered.append('N/A')

            # Extracting product type if available
            try:
                product_element = review.find_element(By.CSS_SELECTOR, 'a[href*=\'stamped.io/go/\']')
                product = product_element.text
                products.append(product)
            except:
                products.append('N/A')

            # Extracting size of product if available
            try:
                fit_element = review.find_element(By.CLASS_NAME, 'stamped-review-option-scale')
                fit = fit_element.get_attribute('data-value')
                fit_rating.append(fit)
            except:
                fit_rating.append('N/A')

            # Extracting height and weight of user if available
            try:
                height_weight_element = review.find_element(By.CSS_SELECTOR, 'li[data-value="what-is-your-height-and-weight"]')
                span_element = height_weight_element.find_element(By.CSS_SELECTOR, 'span[data-value]')
                height_weight = span_element.text
                reviewers_height_and_weight.append(height_weight)
            except:
                reviewers_height_and_weight.append('N/A')
            
        # Check if the "Next page" button is clickable
        try:
            next_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'li.next > a[aria-label="Next page"]'))
            )
        
            # Click the "Next page" button and wait for the new page to load
            driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center', behavior: 'instant'});", next_button)
            driver.execute_script("arguments[0].click();", next_button)
            time.sleep(5)
        except Exception as e:
            print("Error while clicking 'Next page' button:", e)
            break

    return users, review_post_date, location_of_review, star_ratings, review_headers, review_body, fit_rating, products, size_ordered, reviewers_height_and_weight

# Collecting all the urls I'll need to scrape
product_urls = []

# for page_number in range(1, 6):
#     url = f'https://www.ornotbike.com/collections/mens?page={page_number}'

#     driver = webdriver.Chrome()  # Initialize the driver (you may need to adjust the driver path)
#     driver.get(url)

#     # Getting links for all products
#     print('Getting all product links for Page', page_number)

#     find_all_products = driver.find_elements(By.TAG_NAME, 'a')

#     for product in find_all_products:
#         if product.get_attribute('class') == 'product-link':
#             product_url = product.get_attribute('href')
#             if product_url not in product_urls:
#                 product_urls.append(product_url)
#                 print(product_url)

all_users = []
all_users_cleaned = []
all_review_post_date = []
all_review_post_date_cleaned = []
all_location_of_review = []
all_star_ratings = []
all_review_headers = []
all_review_body = []
all_review_body_cleaned = []
all_fit_rating = []
all_product = []
all_sizes_ordered = []
all_reviewers_height_and_weight = []

for url in product_urls:
    users, review_post_date, location_of_review, star_ratings, review_headers, review_body, fit_rating, products, size_ordered, reviewers_height_and_weight = scrape_reviews(url)

    all_users.extend(users)
    all_review_post_date.extend(review_post_date)
    all_location_of_review.extend(location_of_review)
    all_star_ratings.extend(star_ratings)
    all_review_headers.extend(review_headers)
    all_review_body.extend(review_body)
    all_fit_rating.extend(fit_rating)
    all_product.extend(products)
    all_sizes_ordered.extend(size_ordered)
    all_reviewers_height_and_weight.extend(reviewers_height_and_weight)

    print('Users count:', len(all_users))
    print('Date count:', len(all_review_post_date))
    print('Location count:', len(all_location_of_review))
    print('Star count:', len(all_star_ratings))
    print('Header count:', len(all_review_headers))
    print('Body count:', len(all_review_body))
    print('Fit count:', len(all_fit_rating))
    print('Product count:', len(all_product))
    print('Size count:', len(all_sizes_ordered))
    print('Height weight count:', len(all_reviewers_height_and_weight))

data1 = {
    'users': all_users, 
    'date': all_review_post_date, 
    'location': all_location_of_review, 
    'star': all_star_ratings, 
    'headers': all_review_headers, 
    'body': all_review_body, 
    'fit': all_fit_rating, 
    'product': all_product, 
    'size': all_sizes_ordered, 
    'height and weight': all_reviewers_height_and_weight, 
}

df1 = pd.DataFrame(data1)
df1.to_csv('ornotdata_trouble1.csv', encoding = 'utf-8-sig')

# pd.set_option('display.max_columns', None)
# print(df1)

