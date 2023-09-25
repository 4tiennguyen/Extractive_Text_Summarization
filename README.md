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
## A. Data Preparation
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
## B. Function Score
#### a. Title feature (F1)  
It is defined as a ratio of the number of matches of the Title words (Tw) in the current sentence (S) to the number of words (w) of the Title (T)      
F1(Sentence, article title) = (number of title words in the sentence) / (total number of words in the sentence)

Above is an article where each sentence has an individual F1 score. The highest score is red and the lowest score is green. The first sentence has a high score for the article titled ‘labor plans maternity pay rise’ because it contains the words ‘maternity’ ‘pay’ and ‘rise’. The sentences with 0 have no title words.
#### b. Sentence Length (F2)  
It is defined as a ratio of the number of words (w) in the current sentence (S) to the number of words in the longest sentence (LS) in the text.    
F2(Sentence, article) = (total number of words in the sentence) / (total number of words in the longest sentence)  

#### c. Sentence position (F3)  
Measure the position of the sentence within the article  
F3(Sentence, article) = MAX[ (1 / (position of sentence)), ( 1 / ((number of sentences in the article) - (position of sentence) + 1) ]  


#### d.Term Weight (F4)  
It is defined as a ratio of the sum of the frequencies of term occurrences (TO) in a sentence (S) to the sum of the frequency of term occurrences in the text.
F4(Sentence, article) = Σ (term occurrences in the sentence) / Σ (term occurrences in the article)  

#### e. Proper Noun (F5)     
It is defined as a ratio of the number of proper nouns (PN) in a sentence (S) to the length (L) of a sentence  
F5(Sentence) = (number of proper nouns in the sentence) / (total number of words in the sentence)
#### f. Numerical Data (F6)  
 It is defined as a ratio of the number of numerical data (ND) in the sentence (S) to the length (L) of the sentence     
F6(Sentence) = (number of numerical terms in the sentence) / (total number of words in the sentence)  
## C. F Score
Keeping the goal in mind to determine a sentence's overall importance,  we wanted to use the functions described by combining them with a weight and summing them together to get a final F score that we will use to determine if a sentence is important or not. We also divide by the sum of all our weights to keep the F score between 0 and 1. This means we also have a sub goal of determining which functions will be the most useful to predict a sentence's importance.  
**F score = (W1×F1+W2×F2+W3×F3+W4×F4+W5×F5+W6×F6) / (W1+W2+W3+W4+W5+W6)**  
Originally we had an intuitive guess that function 1 - title function would be the most important, then function 5 - the proper noun function, followed by functions 2, 3, and 4 having neutral importance and function 6 with the least importance.  
# Milestone 2
## A. Structure of our Baseline Neural Network
![image](https://github.com/4tiennguyen/Extractive_Text_Summarization/assets/34051678/e8e7ed63-b633-4954-9124-6693d975d677)  
To find what weights we should use for each function we used a neural network whose outputs are the 6 function weights. A sentence is first processed outside of our model and tokenized, then the tokens become the input for our neural network. This configuration is the baseline model that we found through trial and error testing. We have three hidden layers in our neural network, with 500, 300, and 200 neurons in each layer respectively. All the hidden layers have relu activation. The output layer of our model has 6 neurons and uses softmax activation each output is used as a weight for the function score in our custom loss. Function scores are passed through Y when we fit the model as a numpy array where the first column of Y is our targets, and the preceding columns are all the individual function scores so that we can access them when we define our custom loss.  
## B. Target Dataframe
![image](https://github.com/4tiennguyen/Extractive_Text_Summarization/assets/34051678/8cdbace7-7f3f-4a03-8176-40ba485237ed)  
This is our target dataframe. One column has our actual targets and all the other columns correspond to our pre-calcuated F scores. We had to pass these through the neural network in Python as our target data frame so that we could access the scores while inside the neural network.  
Originally we tried saving them as a list and multiplying the list within the network but we quickly realized because samples pass through at random and we use stochastic gradient descent we have mini batches all with different samples we couldn’t find a way to match the right scores with the right weights without passing the scores through the neural network somehow.



