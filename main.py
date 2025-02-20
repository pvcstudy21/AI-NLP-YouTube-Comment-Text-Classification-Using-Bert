import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import time


def classify_text(new_text, model, tokenizer, device):
    model.eval()
    inputs = tokenizer(new_text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
    return prediction


def scrape_youtube_comments(url, driver):
    driver.get(url)
    time.sleep(10)  # Initial load time

    # Scroll to load all comments
    screen_height = driver.execute_script("return window.screen.height;")
    scroll_count = 1
    scroll_pause_time = 5

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
    comments_section = soup.find_all('ytd-comment-thread-renderer')

    # Collect comments in a list
    comments_data = []
    for comment_section in comments_section:
        author_tag = comment_section.find('a', {'id': 'author-text'})
        comment_tag = comment_section.find('yt-attributed-string', {'id': 'content-text'})
        time_tag = comment_section.find('span', {'id': 'published-time-text'})

        if author_tag and comment_tag and time_tag:
            author = author_tag.get_text(strip=True)
            comment_text = comment_tag.get_text(strip=True)
            comment_time = time_tag.get_text(strip=True)
            comments_data.append({
                'Author': author,
                'Comment': comment_text,
                'Time': comment_time
            })

    return comments_data


def classify_and_save(comments, model, tokenizer, device, trolls_csv, fans_csv, all_comments_csv):
    classified_data = []

    print(f"Starting classification of {len(comments)} comments...")

    for i, comment in enumerate(comments, 1):
        try:
            classification = classify_text(comment['Comment'], model, tokenizer, device)
            label = "Troll" if classification == 0 else "Fan"
            classified_data.append({
                'Author': comment['Author'],
                'Comment': comment['Comment'],
                'Time': comment['Time'],
                'Classification': label
            })

            if i % 10 == 0:  # Progress update every 10 comments
                print(f"Classified {i}/{len(comments)} comments")

        except Exception as e:
            print(f"Error classifying comment {i}: {str(e)[:100]}...")

    if classified_data:
        df = pd.DataFrame(classified_data)

        # Save all comments with classifications
        df.to_csv(all_comments_csv, index=False)

        # Save separated files for trolls and fans
        trolls_df = df[df['Classification'] == 'Troll']
        fans_df = df[df['Classification'] == 'Fan']

        trolls_df.to_csv(trolls_csv, index=False)
        fans_df.to_csv(fans_csv, index=False)

        print("\nClassification Summary:")
        print(f"Total comments processed: {len(df)}")
        print(f"Troll comments: {len(trolls_df)} ({len(trolls_df) / len(df) * 100:.1f}%)")
        print(f"Fan comments: {len(fans_df)} ({len(fans_df) / len(df) * 100:.1f}%)")
        print(f"\nResults saved to:")
        print(f"- All comments: {all_comments_csv}")
        print(f"- Troll comments: {trolls_csv}")
        print(f"- Fan comments: {fans_csv}")
    else:
        print("No comments were successfully classified.")


def main():
    # Paths and configurations
    model_path = 'trained_bert_model'
    chromedriver_path = r"C:\Users\Dell\Desktop\Artificial Intelligence\YouTube Comments Text Classification Using Bert\chromedriver-win64\chromedriver.exe"
    youtube_url = "https://www.youtube.com/watch?v=Yy2xXp0UGcM"

    # Output files
    all_comments_csv = 'youtube_all_comments_classified.csv'
    trolls_csv = 'youtube_trolls_comments.csv'
    fans_csv = 'youtube_fans_comments.csv'

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the trained model and tokenizer
    try:
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model.to(device)
        print("Model and tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Set up Selenium WebDriver
    service = Service(chromedriver_path)
    driver = webdriver.Chrome(service=service)

    try:
        # Scrape comments
        print(f"\nScraping comments from: {youtube_url}")
        print("This may take several minutes depending on the number of comments...")
        comments = scrape_youtube_comments(youtube_url, driver)
        print(f"Successfully scraped {len(comments)} comments")

        # Classify comments and save results
        if comments:
            classify_and_save(comments, model, tokenizer, device, trolls_csv, fans_csv, all_comments_csv)
        else:
            print("No comments were scraped. Please check the URL and try again.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.quit()
        print("\nBrowser closed. Process complete.")


if __name__ == "__main__":
    main()
