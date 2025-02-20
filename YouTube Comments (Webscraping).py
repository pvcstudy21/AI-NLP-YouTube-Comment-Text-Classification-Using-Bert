# Import necessary libraries
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
import pandas as pd

# Path to the chromedriver executable (update this as needed)
chromedriver_path = r"C:\Users\Dell\Desktop\Artificial Intelligence\YouTube Comments Text Classification Using Bert\chromedriver-win64\chromedriver.exe"

# Set up Chrome WebDriver service
service = Service(chromedriver_path)

# Initialize the Selenium WebDriver
driver = webdriver.Chrome(service=service)

# URL of the YouTube video
youtube_url = "https://www.youtube.com/watch?v=aEoYKH6i7fY"
driver.get(youtube_url)

# Allow the page to load fully
time.sleep(10)

# Scroll the page to load all comments
scroll_pause_time = 5  # Time to pause between scrolls (adjust based on internet speed)
screen_height = driver.execute_script("return window.screen.height;")  # Get screen height
scroll_count = 1

while True:
    # Scroll down by one screen height
    driver.execute_script(f"window.scrollTo(0, {screen_height * scroll_count});")
    scroll_count += 1
    time.sleep(scroll_pause_time)

    # Check if the end of the page is reached
    scroll_height = driver.execute_script("return document.documentElement.scrollHeight;")
    if (screen_height * scroll_count) > scroll_height:
        break

# Extract the page source for parsing
html_source = driver.page_source
soup = BeautifulSoup(html_source, 'html.parser')

# Extract comments based on the appropriate HTML classes
comments_section = soup.find_all('ytd-comment-thread-renderer')

# Collect comments in a list
data = []
for comment_section in comments_section:
    # Extract author
    author_tag = comment_section.find('a', {'id': 'author-text'})
    if author_tag:
        author = author_tag.get_text(strip=True)

    # Extract comment text
    comment_tag = comment_section.find('yt-attributed-string', {'id': 'content-text'})
    if comment_tag:
        comment_text = comment_tag.get_text(strip=True)

    # Extract time
    time_tag = comment_section.find('span', {'id': 'published-time-text'})
    if time_tag:
        comment_time = time_tag.get_text(strip=True)

    # Ensure all values are captured and append to the data list
    if author and comment_text and comment_time:
        data.append({'Author': author, 'Comment': comment_text, 'Time': comment_time})

# Convert the data into a Pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
#output_csv = 'youtube_comments.csv'
#df.to_csv(output_csv, index=False)
#print(f"Comments successfully saved to '{output_csv}'.")

# Close the WebDriver
driver.quit()
