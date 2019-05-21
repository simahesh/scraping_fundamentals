import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import numpy as np
import matplotlib.pyplot as plt

### 1. Web Scraping
url = 'https://www.amazon.in/OnePlus-Mirror-Black-128GB-Storage/product-reviews/B07DJHV6VZ/ref=dpx_acr_txt?showViewpoints=1'
review_html = requests.get(url).text # not understood by BS
soup = BeautifulSoup(review_html, 'lxml') # this is a structured BS object.

#soup.findAll -takes a dict of tags as input and gets the text in that tag
review_titles = soup.findAll('a', {'class': 'a-size-base a-link-normal review-title a-color-base review-title-content a-text-bold'})
review_texts = soup.findAll('span', {'class': 'a-size-base review-text review-text-content'})
profile_names = soup.findAll('span', {'class': 'a-profile-name'})

#constructing DF for better understanding
df=pd.DataFrame({'cust':pd.Series([x.text for x in profile_names]),
	             'title':pd.Series([x.text for x in review_titles]),
	             'text':pd.Series([x.text for x in review_texts])})

### 2. Text Mining
soup = BeautifulSoup(requests.get("http://ptucha.com/").text)
l1 = soup.find_all(type='disc')
data = [listItem.li.text for listItem in l1]
pubRe = re.compile(r"R.W. Ptucha")
patentRe = re.compile(r"U.S. Patent")
#grab matching data in substitute unicode char with ascii
goodData = [re.sub("\x93|\x94", '""', d).strip() for d in data if pubRe.search(d) or patentRe.search(d)]

#plotting 
yearRe = re.compile("19\d{2} | 20\d{2}")
years = np.array([int(yearRe.search(d).group(0)) for d in data if yearRe.search(d) is not None])
numBins = np.max(years) - np.min(years) + 1
plt.figure(figsize=(12, 6))
plt.hist(years, numBins)
plt.title("# of Dr. Ptucha's Publications, Patents, and Presentations per Year")
plt.annotate("Earned M.S.", xy=(2002, 1), xytext=(2000, 4),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()
