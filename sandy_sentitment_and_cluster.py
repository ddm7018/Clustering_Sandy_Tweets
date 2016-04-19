from textblob import TextBlob
import csv
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import operator

largeStr = ""
count = 1
sentOutput= []
wordcount={}
tweets = []

print "preprocessing: tokenizing and cleaning up data"
with open("sandy_tweets.csv",'rU') as csvfile:
	csvreader = csv.reader(csvfile)
	for row in csvreader:
		#print count
		tweet = row[8]
		
		if row[5] == 'en':
			
			tweet = str(tweet.decode("cp850").encode("ascii","ignore"))
			tweetBlob = TextBlob(tweet)
			sentOutput.append([row[9],row[10],tweetBlob.sentiment.subjectivity,tweetBlob.sentiment.polarity])
			tokenizer = RegexpTokenizer(r'\w+')
			tokens = tokenizer.tokenize(tweet)
			tweets.append(tweet)	
			count = count + 1 
			for word in tokens:
				word = word.lower()
				if word not in wordcount:
					wordcount[word] = 1
    			else:
        			wordcount[word] += 1
        	
			
			
c = csv.writer(open("Sent.csv","wb"))
c.writerow(["Lat","Long","Subjectivity","Polarity","Cluster"])

print "finished preprocessing"
print "starting the clustering"

true_k = 7
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(tweets)
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print "Cluster %d:" % i,
    for ind in order_centroids[i, :20]:
        print ' %s' % terms[ind],
    print


print "finished clustering---outputing to csv file"
count = 0
for row in sentOutput:
	row.append(model.labels_[count])
	c.writerow(row)
	count = count +1 
	
print "assembling clusterDict"	

clusterDict =  {}
for ele in range(true_k):
	clusterDict[ele] = []
count = 0 
for row in sentOutput:
	clusterDict[row[4]].append(tweets[count])
	count = count + 1
	
	
		

wc = sorted(wordcount.items(), key=operator.itemgetter(1))




