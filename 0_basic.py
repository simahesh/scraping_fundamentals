import requests
import pandas as pd
from bs4 import BeautifulSoup

url = 'https://www.amazon.in/OnePlus-Mirror-Black-128GB-Storage/product-reviews/B07DJHV6VZ/ref=dpx_acr_txt?showViewpoints=1'
review_html = requests.get(url).text # not understood by BS
soup = BeautifulSoup(review_html, 'lxml') # this is a BS object.

#soup.findAll -takes a dict of tags as input and gets the text in that tag
review_titles = soup.findAll('a', {'class': 'a-size-base a-link-normal review-title a-color-base review-title-content a-text-bold'})
review_texts = soup.findAll('span', {'class': 'a-size-base review-text review-text-content'})
profile_names = soup.findAll('span', {'class': 'a-profile-name'})

#constructing DF for better understanding
df=pd.DataFrame({'cust':pd.Series([x.text for x in profile_names]),
	             'title':pd.Series([x.text for x in review_titles]),
	             'text':pd.Series([x.text for x in review_texts])})
