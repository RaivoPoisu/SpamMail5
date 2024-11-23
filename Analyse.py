from collections import Counter
import pandas as pd
def FindWordCounts(SpamData):
    SpamData["arr"]= SpamData["text"].apply(lambda x: x.split(" "))
    all_words = [word for sublist in SpamData[SpamData["label"]==1]['arr'] for word in sublist]  # Flatten the lists
    word_counts = Counter(all_words)

# Convert the word counts to a DataFrame for better presentation
    word_count_spam_df = pd.DataFrame(word_counts.items(), columns=['word', 'count'])
    word_count_spam_df.sort_values(by='count', ascending=False, inplace=True)
    all_words = [word for sublist in SpamData[SpamData["label"]==0]['arr'] for word in sublist]  # Flatten the lists
    word_counts = Counter(all_words)
    
    # Convert the word counts to a DataFrame for better presentation
    word_count_nospam_df = pd.DataFrame(word_counts.items(), columns=['word', 'count'])
    word_count_nospam_df.sort_values(by='count', ascending=False, inplace=True)
    return word_count_spam_df, word_count_nospam_df