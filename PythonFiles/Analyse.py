from collections import Counter
import pandas as pd
def FindWordCounts(SpamData):
    SpamData["arr"]= SpamData["text"].apply(lambda x: x.split(" "))
    all_words = []
    for sublist in SpamData[SpamData["label"]==1]['arr']:  #This treats the words containing \n as 2 different words
        for word in sublist:
            words = word.split("\n")
            all_words.extend(words)
    word_counts = Counter(all_words)

# Convert the word counts to a DataFrame for better presentation
    word_count_spam_df = pd.DataFrame(word_counts.items(), columns=['word', 'count'])
    word_count_spam_df.sort_values(by='count', ascending=False, inplace=True)
    all_words = []
    for sublist in SpamData[SpamData["label"]==0]['arr']:      #This treats the words containing \n as 2 different words
        for word in sublist:
            words = word.split("\n")
            all_words.extend(words)
    word_counts = Counter(all_words)
    
    # Convert the word counts to a DataFrame for better presentation
    word_count_nospam_df = pd.DataFrame(word_counts.items(), columns=['word', 'count'])
    word_count_nospam_df.sort_values(by='count', ascending=False, inplace=True)
    return word_count_spam_df, word_count_nospam_df