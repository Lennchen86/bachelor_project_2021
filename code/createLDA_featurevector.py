from pprint import pprint
from boltons import iterutils
from gensim.models import CoherenceModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gensim
import gensim.corpora as corpora

def makeDayDescriptorList(df):
    seq = df['Highest_Occ'].tolist()
    Highest_Occ_List = iterutils.chunked(seq, 48)
    s = [[str(i) for i in j] for j in Highest_Occ_List]
    dayDescriptorList = list(map(''.join, s))
    return dayDescriptorList


# determine time slots by using their index numbers, there might be better way of handling it but for now,
# this is the way that I thought of
def determineSlot(i):
    if i < 14:
        return '1'
    elif i < 18:
        return '2'
    elif i < 22:
        return '3'
    elif i < 28:
        return '4'
    elif i < 34:
        return '5'
    elif i < 38:
        return '6'
    elif i < 42:
        return '7'
    elif i < 48:
        return '8'


def fromGPStoDayDescriptor(df):
    df['date_time'] = df.index
    dt = pd.to_datetime(df['date_time'])
    start_hour = dt.dt.hour
    end_hour = pd.Series(np.where(start_hour == 23, 0, start_hour + 1), index=start_hour.index)
    df['Slot'] = start_hour.astype(str) + '-' + end_hour.astype(str)
    df['start_time'] = start_hour.astype(int)
    for i in range(len(df)):
        s = determineSlot(int(df['start_time'].iloc[i]))
        df['Timeslots'].iloc[i] = s


# Append the labels to three letters each
def append3Labels(list):
    list0 = []
    for idx in range(len(list)):
        # it makes sure that string index is inside the range
        if idx == (len(list) - 2):
            break
        # To make the document string needed for further processing
        list0.append(list[idx] + list[idx + 1] + list[idx + 2] + determineSlot(idx))
    return list0


# This function creates the documents needed for LDA input and a histogram to show occurrences of Location
# Sequence Bags.
def makeDocuments(list):
    test = [[list[i]] for i in range(len(list))]
    del test[-1]
    pd.DataFrame(test).to_csv('day_descriptor.csv', index=False)
    alist = []
    for i in range(len(test)):
        tt = test[i]
        for idx in range(len(tt)):
            alist.append(append3Labels(tt[idx]))
        # Make one histogram from the first day
        if i == 0:
            plt.hist(alist[i], edgecolor='black', linewidth=1.2)
            plt.xticks(rotation=45)
            plt.yticks(range(0, 20, 3))
            plt.xlabel("Bag of Location Sequences")
            plt.ylabel("Occurences")
            plt.title("Histogram for first day")
            plt.show()
    pd.DataFrame(alist).to_csv('documents.csv', index=False)
    return alist


def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return (sent_topics_df)


def compute_coherence_values(dictionary, corpus, texts, limit, start, step):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    perplexity_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary,
                                                random_state=100)
        model_list.append(model)
        perplexity_values.append(model.log_perplexity(corpus))
        coherencemodel = CoherenceModel(model=model, texts=texts, corpus=corpus, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values, perplexity_values


def getonedayfeaturevector(document, df1):
    feature_value_for_a_day = 0
    features = []
    # For every topics from the lda_model
    for topic_number in range(len(df1['Topic_number'])):
        topic = df1['Topics'][topic_number]
        # Going through every documents inside the document list within a day
        for doc_idx in range(len(document)):
            # For every entry inside the topic model outcome
            for topic_idx in range(len(topic)):
                # if the same entry is found
                if document[doc_idx] == topic[topic_idx][0]:
                    # Add it to the variable feature_value_for_a day
                    feature_value_for_a_day += topic[topic_idx][1]
        features.append(feature_value_for_a_day)
        feature_value_for_a_day = 0
    return features


# This function returns a list of feature vectors (in length of the number of topics) for each days
# As the input we feed the list of documents and the obtained lda model
def getfeaturevector(list_of_docs, lda_model):
    df1 = pd.DataFrame(columns=['Topic_number', 'Topics'])
    feature_df = pd.DataFrame(columns=['Day', 'Features'])

    for i, t in lda_model.show_topics(formatted=False, num_topics=lda_model.num_topics,
                                      num_words=len(lda_model.id2word)):
        df1 = df1.append({'Topic_number': i, 'Topics': t}, ignore_index=True)
    pd.DataFrame(df1).to_csv('topic_model_show_outcome.csv', index=False)
    featureVec = [getonedayfeaturevector(d, df1) for d in list_of_docs]
    # Save it to a file
    for idx in range(len(featureVec)):
        feature_df = feature_df.append({'Day': idx, 'Features': featureVec[idx]}, ignore_index=True)
    df2 = feature_df.Features.apply(pd.Series)
    # This is the dataframe where the feature vector is stored in a list
    pd.DataFrame(feature_df).to_csv('feature_vectors.csv', index=False)
    # This is the dataframe where the feature vectors are split to each column each
    pd.DataFrame(df2).to_csv('feature_vectors_update.csv', index=False)
    return featureVec, df2


# LDA loading
def modeling(list_of_docs):
    # Create Dictionary
    dictionary = corpora.Dictionary(list_of_docs)
    # Create Corpus
    texts = list_of_docs
    # Term Document Frequency - for each document we create a dictionary reporting how many
    # words and how many times those words appear
    corpus = [dictionary.doc2bow(text) for text in texts]
    # Running LDA using Bag of words
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=15,
                                           random_state=1, passes=10, iterations=100, alpha=0.01,
                                           per_word_topics=True)
    # number of words can be changed, depending on how many we want to retrieve from the topics
    pprint(lda_model.print_topics(num_words=46))
    # Obtaining the main topic for each review:
    all_topics = lda_model.get_document_topics(corpus)
    all_topics_csr = gensim.matutils.corpus2csc(all_topics)
    all_topics_numpy = all_topics_csr.T.toarray()
    major_topic = [np.argmax(arr) for arr in all_topics_numpy]
    # To obtain the feature vector for future use
    feature_vector, featurevec_df = getfeaturevector(texts, lda_model)
    featurevec_df['major_lda_topic'] = major_topic

    model_list, coherence_values, perplexity_values = compute_coherence_values(dictionary=dictionary, corpus=corpus,
                                                                               texts=list_of_docs, start=2, limit=20,
                                                                               step=2)
    # Show graph for coherence score
    limit = 20
    start = 2
    step = 2
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.legend(['coherence value'], loc='best')
    plt.show()
    return feature_vector, featurevec_df
