# Extractive Text Summarization
In the digital age, we have access to a lot of online information. However, reading all of it takes time. Text summarization can help by shortening the text while still keeping important information. Our project uses fuzzy logic and a neural network to extract key sentences. Specifically, we use neural networks to determine the optimal weights for aggregating different feature scores that indicate the importance of sentences in the BBC dataset. The final outcome of this project is the creation of a neural network model with appropriate loss and metric functions, which will produce the optimal weights for aggregating various NLP feature scores.
# Dataset
The BBC News dataset, originating from the BBC's website, provides news article datasets for use as benchmarks in machine learning research. The original data is processed and compiled into a single CSV file for ease of use. The news title, related text file name, content, and category are all preserved. The dataset includes 2225 items, divided into five distinct areas (technology, sport, politics, business, and entertainment), each with an extracted summary. In this project, we only worked on files related to politics area.
# Overview
### 1. What is Extractive Text Summarization?
Extractive text summarization is a technique in natural language processing (NLP) and information retrieval. It automatically selects and extracts the most important sentences or phrases from a given document or text to create a concise summary. The technique doesn't generate new sentences or content. Instead, it extracts and rearranges existing content to form a summary that retains the essential information from the original text.
### 2. What is Fuzzy Logic?
The sentence ranking stage in text summarization has improved over time by taking into account more features. For instance, within a paragraph, we consider the characteristics of each sentence, such as its location, length, similarity to other sentences, any numerical data, and so on as features. With fuzzy logic, we can assign weights, provide membership functions to each feature, and then calculate total scores for each sentence. We then choose the sentences with the highest scores to include in the summary of the paragraph.  
### 3. Project milestones
#### a. Milestone 1: Combining feature vectors to decide which are representative statements  
We propose a formula that combines the F scores for any given sentence.   
#### b. Milestone 2
We implemented NN to find the optimal weight for each feature in milestone 1

# Milestone 1
# Data Preparation
### 1. Sentence separation:
* Separate title sentence of documents by’\n\n’
* Removed punctuations and newlines (such as “,\n) other than period, question mark and exclamation mark that sometimes cause sentence separation issues
* Inserted space after a period, question mark, and exclamation mark if it was followed by a word or non-digit character (This prevents a number like 4.5 from becoming split as a sentence)
* Merge multiple periods and spaces into one.
* Remove numbers after a period if the character before the period is non-digit (Such as John.2)
### 2. NLP Technique
We tried both stemming and lemmatization on our data set, stemming removes the last few characters from a word to reduce it to the stem of the word, and lemmatization converts the word to a meaningful base. So they both are very similar and we decided to go with lemmatization. We also tokenized the sentences and turned the dataset into a bag of words model which is a matrix whose columns are tokens and each row is a sentence with the elements of the matrix being the counts of the token in the sentence.

### 3. Test/Train Data split
Split the data with an 80/20 ratio. The first 20% of the articles (1 - 83) as the Testing data and the last 80% of the articles as the Training data (84 - 417)
### 4. Function Score
#### a. Title feature (F1)  
It is defined as a ratio of the number of matches of the Title words (Tw) in the current sentence (S) to the number of words (w) of the Title (T)      
F1(Sentence, article title) = (number of title words in the sentence) / (total number of words in the sentence)

#### b. Sentence Length (F2)  
It is defined as a ratio of the number of words (w) in the current sentence (S) to the number of words in the longest sentence (LS) in the text.     
<img alt="F2" src="https://drive.google.com/uc?export=view&id=1b42IWpr5svaCyQl6fdnGNWaLP3Rv4bgK">  
#### c. Sentence position (F3)  
It is defined as a maximum of the next two relations      
<img alt="F3" src="https://drive.google.com/uc?export=view&id=1WNqhFyRSkFR7RaGqtZaywOmQVBrUJyb0">  
#### d.Term Weight (F4)  
It is defined as a ratio of the sum of the frequencies of term occurrences (TO) in a sentence (S) to the sum of the frequency of term occurrences in the text.     
<img alt="F4" src="https://drive.google.com/uc?export=view&id=1wS8NV1otHFjgu4NePxkyPwQXFpZ4f93q">  
#### e. Proper Noun (F5)     
It is defined as a ratio of the number of proper nouns (PN) in a sentence (S) to the length (L) of a sentence  
<img alt="F6" src="https://drive.google.com/uc?export=view&id=1MEBknA87K-vPSlu061F-hOz_c9aHrl6W">  
#### f. Numerical Data (F6)  
 It is defined as a ratio of the number of numerical data (ND) in the sentence (S) to the length (L) of the sentence     
<img alt="F6" src="https://drive.google.com/uc?export=view&id=1MEBknA87K-vPSlu061F-hOz_c9aHrl6W">

