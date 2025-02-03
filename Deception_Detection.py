import os,glob,re
from string import punctuation
from collections import defaultdict
from collections import Counter
import numpy as np
from collections import OrderedDict 
import copy
from sklearn import metrics
import operator
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
import pprint
import nltk
from nltk import pos_tag, word_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')

list_negative_word = []
list_positive_word = []
Unique_list_positive_word = []
Unique_list_negative_word = []
cwd = os.getcwd() 
count_positive_ID = 0
count_negative_ID = 0
Negative_Word_Dict = defaultdict()
Positive_Word_Dict = defaultdict()
Sentence_Sentiment = {}
Evaluation_Dict = OrderedDict()



#Count of number of Sentiment Words seem equal!
def count_sentiment_words(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read().lower()
            
        # Simple sentiment analysis using word counts
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'love', 'perfect', 'fantastic'}
        negative_words = {'bad', 'poor', 'terrible', 'awful', 'worst', 'hate', 'horrible', 'disappointing'}
        
        words = word_tokenize(text)
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        # Calculate sentiment score (-1 to 1)
        total = positive_count + negative_count
        if total == 0:
            return 0
        return (positive_count - negative_count) / total
        
    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        return 0  # Neutral sentiment in case of error

def POS_Tagging(sentence):
    try:
        # Use TextBlob for POS tagging
        blob = TextBlob(sentence)
        
        # Count verbs and nouns
        count_verbs = len([word for word, tag in blob.tags if tag.startswith('VB')])
        count_nouns = len([word for word, tag in blob.tags if tag.startswith('NN')])
        
        # Determine if the text might be deceptive based on verb/noun ratio
        if count_verbs > count_nouns:
            return 'F'  # Potentially deceptive
        else:
            return 'T'  # Likely truthful
            
    except Exception as e:
        print(f"Error in POS_Tagging: {str(e)}")
        return 'T'  # Default to truthful in case of error

def calc_Unigram_Probability():
    Train_Set_NEG  = []
    count_neg = 0
    Train_Set_POS  = []
    count_pos = 0
    Training_Set_POS = []
    Training_Set_NEG = []
    NEG_Tags = []
    POS_Tags = []
    ID_val_List  = []
    Neg_Tags_Dict = OrderedDict()
    Pos_Tags_Dict = OrderedDict()
    
    for line in open("hotelF-train.txt",encoding="utf8").readlines():
        if line.strip():
            word_split = line.split()
            word_split = line.replace(',',' ').replace('.',' ').replace('!',' ').replace('--',' ').split()
            for each_word in word_split:
                if re.match(r'ID-[0-9].*',each_word):
                    Id_Value = each_word
                    ID_val_List.append(Id_Value)
                    continue
                if each_word not in frequent_word_list:
                    Train_Set_NEG.append(each_word)
            text = nltk.word_tokenize(line)
            try:
                tagged_list = nltk.pos_tag(text)
                tags = [x[1] for x in tagged_list]
                for each_item in tags:
                    if each_item not in ['.',',','!','--']:
                        NEG_Tags.append(each_item)
                Neg_Tags_Dict.update({Id_Value:NEG_Tags})
                NEG_Tags = []
            except Exception as e:
                print(f"Error in POS tagging: {str(e)}")
                continue

    for line in open("hotelT-train.txt",encoding="utf8").readlines():
        if line.strip():
            word_split = line.split()
            word_split = line.replace(',',' ').replace('.',' ').replace('!',' ').replace('--',' ').split()
            for each_word in word_split:
                if re.match(r'ID-[0-9].*',each_word):
                    Id_Value = each_word
                    ID_val_List.append(Id_Value)
                    continue
                if each_word not in frequent_word_list:
                    Train_Set_POS.append(each_word)
            text = nltk.word_tokenize(line)
            try:
                tagged_list = nltk.pos_tag(text)
                tags = [x[1] for x in tagged_list]
                for each_item in tags:
                    if each_item not in ['.',',','!','--']:
                        POS_Tags.append(each_item)
                Pos_Tags_Dict.update({Id_Value:POS_Tags})
                POS_Tags = []
            except Exception as e:
                print(f"Error in POS tagging: {str(e)}")
                continue

    return Pos_Tags_Dict, Neg_Tags_Dict

frequent_word_list = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
ID_Val = []
#Basic data cleaning using regular expressions. I have removed the frequently repeated words
#as described in the above list. I have eliminated all kind of punctuation marks - like .,!,-, etc
for line in open("hotelF-train.txt",encoding="utf8").readlines():
    if line.strip():
        word_split = line.replace(',',' ').replace('.',' ').replace('!',' ').replace('--',' ').split()
        for each_word in word_split:
                        if re.match(r'[IDidIdiD].*-[0-9].*',each_word):
                            count_negative_ID+=1
                            ID_Val.append(each_word)
                            continue
                        elif re.match(r'(\d+)\.(\d+)+',each_word):
                            continue
                        elif re.match(r'(\d+)+',each_word):
                            continue 
                        elif re.match(r'$(\d+)+',each_word):
                            continue  
                        elif re.match(r'[A-Za-z]*\.+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'[A-Za-z]*\?+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'[A-Za-z]*\!+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'-+',each_word):
                            continue
                        elif re.match(r'[A-Za-z]*\)+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'\(+[A-Za-z]*',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'[A-Za-z]*\,+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'[A-Za-z]*\/+[A-Za-z]*',each_word):
                            list_word = each_word.split('/')
                            if list_word[0] not in frequent_word_list:
                                list_negative_word.append(list_word [0])
                            else:
                                continue
                            
                            if list_word[1] not in frequent_word_list:
                                list_negative_word.append(list_word [1])
                            else:
                                continue
                        if each_word in frequent_word_list:
                            continue
                        else:
                            list_negative_word.append(each_word)

#for each_item in ID_Val:
#    Evaluation_Dict.update({each_item:'F'})

List_Size_Negative = len(list_negative_word)
Negative_Word_Dict =  Counter(list_negative_word)
Denominator_Sum_Neg = sum(Negative_Word_Dict.values())
Unique_list_negative_word = set(list_negative_word)
Unique_List_Size_Negative = len(Unique_list_negative_word)


ID_Val2 = []
list_positive_word = []
for line in open("hotelT-train.txt",encoding="utf8").readlines():
    if line.strip():
        word_split = line.split()
        word_split = line.replace(',',' ').replace('.',' ').replace('!',' ').replace('--',' ').split()
        for each_word in word_split:
                        if re.match(r'ID-[0-9].*',each_word):
                            count_positive_ID+=1
                            ID_Val2.append(each_word)
                            continue
                        elif re.match(r'(\d+)\.(\d+)+',each_word):
                            continue
                        elif re.match(r'(\d+)+',each_word):
                            continue 
                        elif re.match(r'$(\d+)+',each_word):
                            continue  
                        elif re.match(r'[A-Za-z]*\.+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'[A-Za-z]*\?+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'[A-Za-z]*\!+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'-+',each_word):
                            continue
                        elif re.match(r'[A-Za-z]*\)+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'\(+[A-Za-z]*',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'[A-Za-z]*\,+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'[A-Za-z]*\/+[A-Za-z]*',each_word):
                            list_word = each_word.split('/')
                            if list_word[0] not in frequent_word_list:
                                list_positive_word.append(list_word [0])
                            else:
                                continue
                            
                            if list_word[1] not in frequent_word_list:
                                list_positive_word.append(list_word [1])
                            else:
                                continue
                        if each_word in frequent_word_list:
                            continue
                        else:
                            list_positive_word.append(each_word)

#for each_item in ID_Val2:
 #   Evaluation_Dict.update({each_item:'T'})
#pprint.pprint (Evaluation_Dict)

 
#List of all words in positive training set with repitition
List_Size_Positive = len(list_positive_word)
#Dictionary having the words : count 
Positive_Word_Dict = Counter(list_positive_word)
Denominator_Sum_Pos = sum(Positive_Word_Dict.values())

#List of all unique words in Positive Training set
Unique_list_positive_word = set(list_positive_word)
#Length of Unique Size list
Unique_List_Size_Positive = len(Unique_list_positive_word)
Total_No_of_Docs = count_positive_ID + count_negative_ID
print ("count_positive_ID - ",count_positive_ID)
print ("count_negative_ID - ",count_negative_ID)
print ("Total_No_of_Docs - ",Total_No_of_Docs)

Log_Prior_Positive = np.log10(float(count_positive_ID)/float(Total_No_of_Docs))
Log_Prior_Negative = np.log10(float(count_negative_ID)/float(Total_No_of_Docs))
print ("Log_Prior_Positive - ",Log_Prior_Positive)
print ("Log_Prior_Negative - ",Log_Prior_Negative)

Negative_Word_Dict_final = defaultdict()
Positive_Word_Dict_final = defaultdict()

print (Denominator_Sum_Pos)
print (Denominator_Sum_Neg)
#Creating maximum likelihood estimate values for each word in positive training set
for key in Positive_Word_Dict:    
    val = np.log10 (float(Positive_Word_Dict[key] + 1)/float(Denominator_Sum_Pos + Unique_List_Size_Positive))
    Positive_Word_Dict_final.update({key:val})
for key in Negative_Word_Dict:    
    val =  np.log10(float(Negative_Word_Dict[key] + 1)/float(Denominator_Sum_Neg + Unique_List_Size_Negative))
    Negative_Word_Dict_final.update({key:val})

Negative_Probability_Sentence = 0.00
Positive_Probability_Sentence = 0.00
Sentence_Token_List = []
Sentence_List = []
POS_Tag_Prob = OrderedDict()
NEG_Tag_Prob = OrderedDict()
POS_Tag_Prob,NEG_Tag_Prob = calc_Unigram_Probability()

with open("hotelDeceptionTest.txt",'r',encoding="utf8") as f:
   token =  [line.split() for line in f]
   for each_word in token:
        Sentence_Token_List.append(each_word)

Test_Review_Dict = OrderedDict()
Test_Review_Class = OrderedDict()
temp_dict = {}
List_Tokens_ID = []
temp = []

for i in range (0,len(Sentence_Token_List)):
    for j in range (0,len(Sentence_Token_List[i])):
        word = Sentence_Token_List[i][j]
        if re.match(r'ID-[0-9].*',word):
            Id_Value = word
            continue
        elif re.match(r'(\d+)\.(\d+)+',word):
            continue
        elif re.match(r'(\d+)+',word):
            continue 
        elif re.match(r'$(\d+)+',word):
            continue  
        elif re.match(r'[A-Za-z]*\.+',word):
            word = re.sub(r'[^\w\s]','',word)
        elif re.match(r'[A-Za-z]*\?+',word):
            word = re.sub(r'[^\w\s]','',word)
        elif re.match(r'[A-Za-z]*\!+',word):
            word = re.sub(r'[^\w\s]','',word)
        elif re.match(r'-+',word):
            continue
        elif re.match(r'[A-Za-z]*\)+',word):
            word = re.sub(r'[^\w\s]','',word)
        elif re.match(r'\(+[A-Za-z]*',word):
            word = re.sub(r'[^\w\s]','',word)
        elif re.match(r'[A-Za-z]*\,+',word):
            word = re.sub(r'[^\w\s]','',word)
        elif re.match(r'[A-Za-z]*\/+[A-Za-z]*',word):
            list_word = word.split('/')
            if list_word[0] not in frequent_word_list:
                List_Tokens_ID.append(list_word[0])
            else:
                continue
                            
            if list_word[1] not in frequent_word_list:
                List_Tokens_ID.append(list_word[1])
            else:
                continue
        if (word not in frequent_word_list):
                List_Tokens_ID.append(word)
    temp = copy.copy(List_Tokens_ID)
    Test_Review_Dict.update ({Id_Value:temp})
    List_Tokens_ID[:] = []

Sentiment_Word_list = []
for line in open("Sentiment_Word_List.txt",encoding="utf8").readlines():
    if line.strip():
        word_split = line.split()
        Sentiment_Word_list.append(word_split[0])

for each_key,each_value in Test_Review_Dict.items():
    ID_Value = each_key
    for val in each_value:
        if ((val in Negative_Word_Dict_final) and (val in Positive_Word_Dict_final)):
            Negative_Probability_Sentence += Negative_Word_Dict_final[val]
            Positive_Probability_Sentence += Positive_Word_Dict_final[val]

    Sentence =  ' '.join(map(str, each_value))
    text=nltk.word_tokenize(Sentence)
    Tagged_Text = (nltk.pos_tag(text))
    for i in Tagged_Text:
        if i[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ','NN', 'NNS', 'NNP', 'NNPS']:
            if (i[1] in POS_Tag_Prob) and (i[1] in NEG_Tag_Prob):
                Positive_Probability_Sentence+= POS_Tag_Prob[i[1]]
                Negative_Probability_Sentence+= NEG_Tag_Prob[i[1]]
        else:
            continue

    Positive_Probability_Sentence+=Log_Prior_Positive
    Negative_Probability_Sentence+=Log_Prior_Negative
    temp_dict.update({'T': Positive_Probability_Sentence})
    temp_dict.update({'F': Negative_Probability_Sentence})
    #print ("ID- Value : {}, False : {} , True : {}".format(ID_Value, Negative_Probability_Sentence,Positive_Probability_Sentence))
    key = [k for k,v in temp_dict.items() if v==max(temp_dict.values())][0] 
    Test_Review_Class.update ({ID_Value:key})
    temp_dict.clear()
    Positive_Probability_Sentence = 0.0
    Negative_Probability_Sentence = 0.0

    
with open("Output.txt",'w') as ofile:
        for keys,values in Test_Review_Class.items():
            ofile.write((str(keys) + '\t' + values + '\n'))


def NaiveBayes_BuiltIn_Package():
    Train_Set_NEG  = []
    count_neg = 0
    Training_Set = []
    ID_val_List  = []
    for line in open("hotelF-train.txt",'r',encoding="utf8").readlines():
        if re.match(r'ID-[0-9].*',line):
            count_neg+=1
            each_line = re.sub(r'ID-[0-9].*_','',line)
            each_line = each_line.strip("\n")
            Train_Set_NEG.append(each_line[8:])
    
    for each_sentence in Train_Set_NEG:
        Training_Set.append((each_sentence,'F'))
    
    Train_Set_POS  = []
    count_pos = 0
    for line in open("hotelT-train.txt",'r',encoding="utf8").readlines():
        if re.match(r'ID-[0-9].*',line):
            count_pos+=1
            each_line = re.sub(r'ID-[0-9].*_','',line)
            each_line = each_line.strip()
            Train_Set_POS.append(each_line[8:])

    for each_sentence in Train_Set_POS:
        Training_Set.append((each_sentence,'T'))

    classifier = NaiveBayesClassifier(Training_Set)  


#pprint.pprint (Evaluation_Dict)
#pprint.pprint (Test_Review_Class)
#pprint.pprint (Test_Review_Dict)
#pprint.pprint (POS_Tag_Prob)
#pprint.pprint (NEG_Tag_Prob)
#pprint.pprint (Positive_Word_Dict_final)
#pprint.pprint (Negative_Word_Dict_final)
#pprint.pprint (Positive_Probability_Sentence)
#pprint.pprint (Negative_Probability_Sentence)
#pprint.pprint (Log_Prior_Positive)
#pprint.pprint (Log_Prior_Negative)
#pprint.pprint (count_positive_ID)
#pprint.pprint (count_negative_ID)
#pprint.pprint (Total_No_of_Docs)
#pprint.pprint (Denominator_Sum_Neg)
#pprint.pprint (Denominator_Sum_Pos)
#pprint.pprint (Unique_List_Size_Positive)
#pprint.pprint (Unique_List_Size_Negative)
#pprint.pprint (List_Size_Positive)
#pprint.pprint (List_Size_Negative)
#pprint.pprint (Negative_Word_Dict)
#pprint.pprint (Positive_Word_Dict)
#pprint.pprint (Unique_list_negative_word)
#pprint.pprint (Unique_list_positive_word)
#pprint.pprint (list_negative_word)
#pprint.pprint (list_positive_word)
#pprint.pprint (ID_Val)
#pprint.pprint (ID_Val2)
#pprint.pprint (frequent_word_list)
#pprint.pprint (Evaluation_Dict)
#pprint.pprint (Test_Review_Dict)
#pprint.pprint (Test_Review_Class)
#pprint.pprint (POS_Tag_Prob)
#pprint.pprint (NEG_Tag_Prob)
#pprint.pprint (Positive_Word_Dict_final)
#pprint.pprint (Negative_Word_Dict_final)
#pprint.pprint (Positive_Probability_Sentence)
#pprint.pprint (Negative_Probability_Sentence)
#pprint.pprint (Log_Prior_Positive)
#pprint.pprint (Log_Prior_Negative)
#pprint.pprint (count_positive_ID)
#pprint.pprint (count_negative_ID)
#pprint.pprint (Total_No_of_Docs)
#pprint.pprint (Denominator_Sum_Neg)
#pprint.pprint (Denominator_Sum_Pos)
#pprint.pprint (Unique_List_Size_Positive)
#pprint.pprint (Unique_List_Size_Negative)
#pprint.pprint (List_Size_Positive)
#pprint.pprint (List_Size_Negative)
#pprint.pprint (Negative_Word_Dict)
#pprint.pprint (Positive_Word_Dict)
#pprint.pprint (Unique_list_negative_word)
#pprint.pprint (Unique_list_positive_word)
#pprint.pprint (list_negative_word)
#pprint.pprint (list_positive_word)
#pprint.pprint (ID_Val)
#pprint.pprint (ID_Val2)
#pprint.pprint (frequent_word_list)
