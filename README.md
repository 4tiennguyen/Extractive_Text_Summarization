# Extractive Text Summarization
In the digital age, we have access to a lot of online information. However, reading all of it takes time. Text summarization can help by shortening the text while still keeping important information. Our project uses fuzzy logic and a neural network to extract key sentences. Specifically, we use neural networks to determine the optimal weights for aggregating different feature scores that indicate the importance of sentences in the BBC dataset. The final outcome of this project is the creation of a neural network model with appropriate loss and metric functions, which will produce the optimal weights for aggregating various NLP feature scores.
# Dataset
The BBC News dataset, originating from the BBC's website, provides news article datasets for use as benchmarks in machine learning research. The original data is processed and compiled into a single CSV file for ease of use. The news title, related text file name, content, and category are all preserved. The dataset includes 2225 items, divided into five distinct areas (technology, sport, politics, business, and entertainment), each with an extracted summary. In this project, we only worked on files related to politics area.
# Overview
### 1. What is Extractive Text Summarization?
Extractive text summarization is a technique in natural language processing (NLP) and information retrieval. It automatically selects and extracts the most important sentences or phrases from a given document or text to create a concise summary. The technique doesn't generate new sentences or content. Instead, it extracts and rearranges existing content to form a summary that retains the essential information from the original text.
### 2. What is Fuzzy Logic?
The sentence ranking stage in text summarization has improved over time by taking into account more features. For instance, within a paragraph, we consider the characteristics of each sentence, such as its location, length, similarity to other sentences, any numerical data, and so on as features. With fuzzy logic, we can assign weights, provide membership functions to each feature, and then calculate total scores for each sentence. We then choose the sentences with the highest scores to include in the summary of the paragraph.  
a. Title feature (F1)  
It is defined as a ratio of the number of matches of the Title words (Tw) in the current sentence (S) to the number of words (w) of the Title (T)     
<img alt="F1" src="https://drive.google.com/uc?export=view&id=1J1GGrkuGcHhPlGRIo7xmPlwHLe0I2LcV">  
b. Sentence Length (F2)  
It is defined as a ratio of the number of words (w) in the current sentence (S) to the number of words in the longest sentence (LS) in the text.     
<img alt="F2" src="https://drive.google.com/uc?export=view&id=1b42IWpr5svaCyQl6fdnGNWaLP3Rv4bgK">  
c. Sentence position (F3)  
It is defined as a maximum of the next two relations      
<img alt="F3" src="https://drive.google.com/uc?export=view&id=1WNqhFyRSkFR7RaGqtZaywOmQVBrUJyb0">  
d.Term Weight (F4)  
It is defined as a ratio of the sum of the frequencies of term occurrences (TO) in a sentence (S) to the sum of the frequency of term occurrences in the text.     
<img alt="F4" src="https://drive.google.com/uc?export=view&id=1wS8NV1otHFjgu4NePxkyPwQXFpZ4f93q">  
e. Proper Noun (F5)     
It is defined as a ratio of the number of proper nouns (PN) in a sentence (S) to the length (L) of a sentence  
<img alt="F6" src="https://drive.google.com/uc?export=view&id=1MEBknA87K-vPSlu061F-hOz_c9aHrl6W">  
f. Numerical Data (F6)  
 It is defined as a ratio of the number of numerical data (ND) in the sentence (S) to the length (L) of the sentence     
<img alt="F6" src="https://drive.google.com/uc?export=view&id=1MEBknA87K-vPSlu061F-hOz_c9aHrl6W">
### 3. Project milestones
##### a. Milestone 1: Combining feature vectors to decide which are representative statements  
We propose a formula that combines the F scores for any given sentence.   
##### b. Milestone 2
We implemented NN to find the optimal weight for each feature in milestome 1
