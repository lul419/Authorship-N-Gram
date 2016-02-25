import sys
import urllib.request
import re
import math
import random


def dev(author_list):

	smooth_method = ["G_T","laplace"]
	author_LM_dict = {}
	dev_dict_for_all_authors = {}
	N_gram = {"Dickens":3,"Christie":3,"Austen":3,"Doyle":3,"Thoureau":3,"Whitman":3,"Wilde":3}
	devSetLengthDict = {}
	perplex_report= {}
	linear_interpolation_lambda = {}
	for author in author_list.keys():
		print(author)
			# text_of_author := a list of tokens from a author'set work. 
		text_of_author = author_list[author]
			# create the development set and training set for each author and clean the url data. dev_set = [s_1,s_2,...,s_50]
		cleaned_text, dev_set, training_set = clean_Data_And_Get_Dev_Training_Set(text_of_author)
			# get all N_grams counts and v
		unigram, bigram, trigram, quadgram, quintgram, v =  N_gram_count(training_set)
			# add to the author_LM_dict
		LM_list = [unigram, bigram, trigram, quadgram, quintgram, v]
		author_LM_dict[author] = [unigram, bigram, trigram, quadgram, quintgram, v]
			# add the author's dev set to the dev dictz for all authors. 
		dev_dict_for_all_authors[author] = dev_set
			# get the perplexity report of the performance of every language model for the given author
		perplex_report[author] = perplexity_report(LM_list,dev_set[3:5],smooth_method[0])
			# store the length of each dev set
		devSetLengthDict[author] = len(dev_set)

		#best_lamda = get_best_lamda(dev_set,N_gram.get(author),LM_list,smooth_method[0])
		#linear_interpolation_lambda[author] = best_lamda
 
		# for the first test, we let austen to use trigram with laplace, christie with laplace and dickens with laplace
	correctness_dictionary = test_on_dev(author_LM_dict, dev_dict_for_all_authors, N_gram)
		# print the dev result out
	print_print_dict(correctness_dictionary,devSetLengthDict)
		# write the perplexity report to a file
	write_perplexity_report(perplex_report)
	#write_best_Linear_Interpolation(linear_interpolation_lambda)

def write_best_Linear_Interpolation(linear_interpolation_lambda):
	lambda_set_file = open("linear_interpolation_lambda.txt","w")
	for author in linear_interpolation_lambda.keys():
		best = linear_interpolation_lambda.get(author)
		lambda_set_file.write(author)
		lambda_set_file.write(" | ")
		lambda_set_file.write(str(best))
		lambda_set_file.write("\n")
	lambda_set_file.close()
	return 


def clean_line(line):
	one_sentence = ["<s>"]
		
		# put whitespaces around unambiguous separators
	line = re.sub(r'([\\?!()\";/\\|])',r' \1 ',line)
		# put whitespaces around comma that are not inside numbers. 
	line = re.sub(r'([^0-9]),',r'\1 , ',line)
	line = re.sub(r',([^0-9])',r' , \1',line)

		# distinguish singlequotes from apostrophes by segmenting off single quotes not preceded by letter
	line = re.sub(r"^'",r" ' ",line)
	line = re.sub(r"([^a-zA-Z0-9])'",r"\1 ' ",line)

		# segment off unambiguous word-final punctuations.
	line = re.sub(r"('|:|-)",r" \1",line)
	line = re.sub(r"('|:|-)([^a-zA-Z])",r" \1 \2",line)

	abbrevList = ['Mr.','Ms.','Mrs.','Doc.']

	# deal with periods. 

	# for every word in the book.
	for word in line.split(" "):
		if re.match(r".*\.",word):
			if re.match(r"[a-zA-Z0-9]+\.",word) and \
			not re.match(r'([a-zA-Z]\.[a-zA-Z]\.)+|[A-Z][bcdfghj-nptvxz]+\.',word) and word not in abbrevList:
				word = re.sub(r'\.',r' \.',word)
		if re.match(r"^[a-zA-Z0-9]",word):
			chars = word.split(" ")
			for char in chars:
				if re.match(r"[a-zA-Z0-9]+",char):	
					one_sentence.append(char.lower())
	one_sentence.append("<'s'>")

	return one_sentence
def test(test_file,author_list):
	test_file = open(test_file)
	smooth_method = ["G_T","laplace"]
	author_LM_dict = {}
	line_predict_dict = {}
	N_gram = {"Dickens":1,"Christie":1,"Austen":2,"Doyle":1,"Thoureau":1,"Whitman":1,"Wilde":2}
	lambda_parameter = {"Dickens":[1],"Christie":[1],"Austen":[0.324,0.676],"Doyle":[1],"Thoureau":[1],"Whitman":[1],"Wilde":[0.446,0.554]}
	devSetLengthDict = {}
	linear_int = True
	for author in author_list.keys():
	
		text_of_author = author_list[author]
		cleaned_text, dev_set, training_set = clean_Data_And_Get_Dev_Training_Set(text_of_author)
		unigram, bigram, trigram, quadgram, quintgram, v =  N_gram_count(training_set)
		author_LM_dict[author] = [unigram, bigram, trigram, quadgram, quintgram, v]

	for line in test_file:
		cleaned_line = clean_line(line.strip("\n"))
		
		optimal_prob = float("-inf")
		prediction = None
		for author_ in N_gram.keys():
			if linear_int:
				lambda_set = lambda_parameter.get(author_)
				prob = dev_set_prob_linear_interpolation(cleaned_line,N_gram.get(author_),lambda_parameter.get(author_),author_LM_dict.get(author_),smooth_method[0])
			else:
				prob = nGram_with_smoothing(N_gram.get(author_), cleaned_line, author_LM_dict.get(author_), smooth_method[0])

			if prob > optimal_prob:
				optimal_prob = prob
				prediction = author_
		line_predict_dict[str(line)] = prediction
		print(optimal_prob)
		print(prediction)
	return line_predict_dict


# perplexity functions

def perplexity_report(LM_list, devSet, smooth_method):
	union_dev = []
	perplexity_measure = {}
	N = 0
	best_per = float("-inf")
	best_LM = None
	for sent in devSet:
		for word in sent:
			union_dev.append(word)
			if word != "<s>":
				N = N + 1
		# for every language model, calculate its perplexity over the development set. 
	i = 1
	while i < len(LM_list):
		perplexity = calculate_perplexity(i,union_dev,N,LM_list,smooth_method)
		perplexity_measure[i] = perplexity
		if perplexity > best_per:
			best_per = perplexity
			best_LM = i
		i = i + 1

	return perplexity_measure 
def calculate_perplexity(i,union_dev,N,LM_list,smooth_method):
	probability = nGram_for_perplexity(i,union_dev,LM_list,smooth_method)
	perplexity = probability ** (1/N)
	return perplexity
# return a dictionary of correct predications of each author's dev set. 
def non_log_smooth_method_prob(prob, count1, count2, smoothMethod, v, n_Gram):
	k = 5
	if smoothMethod == "G_T":
		N_c_dict = get_N_c_dict(n_Gram)
		if count1 > k:
			prob = (1/(count1/count2)) * prob
		elif 0 < count1 <= k:
			N_c1 = N_c_dict[count1+1]
			N_c = N_c_dict[count1]
			N_k1 = N_c_dict[k+1]
			N_k =  N_c_dict[k]
			N_1 = N_c_dict[1]
			discount_factor = ((count1+1)*(N_c1/N_c) - (((k+1)*count1*N_k1)/N_1))/(1-(((k+1)*N_k1)/N_1))
			prob = (1/((count1/count2)*discount_factor)) * prob
		elif count1 == 0 :
			gained_distribution = (N_c_dict[1]/N_c_dict[0]) / N_c_dict[0]
			prob = (1/gained_distribution) * prob
	elif smoothMethod == "laplace":
		prob = (1/((count1 + 1)/ (count2 + v))) * prob
	else:
		if count1 != 0 and count2 != 0:
			prob = (1/(count1/ count2)) * prob
		else:
			print("both denominator and numerator are zero.")
	return prob
def nGram_for_perplexity(n,word_list,LM_list,smooth_method):
	v = LM_list[5] 
	Prob = 1
	if n == 1:
		total = 0
		n_Gram = LM_list[0]
		for value in n_Gram.values():
			total = total + value		
		for i in range(len(word_list)):
			key = "["+"'"+word_list[i] + "'" + "]"
			if LM_list[0].get(key) == None:
				numerator_count = 0
			else:
				numerator_count = LM_list[0].get(key)
			Prob = non_log_smooth_method_prob(Prob,numerator_count,total, smooth_method,v,n_Gram)
	else:
		n_Gram = LM_list[n-1] 
		n_minus_1_Gram = LM_list[n-2]
		for i in range(len(word_list)):
			if i >= n-1:
				numerator = str(word_list[i-(n-1):i+1])
				denominator = str(word_list[i-(n-1):i])
				if numerator in n_Gram.keys():
					numerator_count = n_Gram.get(numerator)
				else:
					numerator_count = 0
				if n_minus_1_Gram.get(denominator) != None:	
					denominator_count = n_minus_1_Gram.get(denominator)
				else:
					denominator_count = 0
				Prob = non_log_smooth_method_prob(Prob, numerator_count,denominator_count, smooth_method, v, n_Gram)
			
	return Prob
def write_perplexity_report(perplexity_report):
	perplexity_measure_file = open("perplexity_measure.txt","w")
	for author in perplexity_report.keys():
		perplexity_measure = perplexity_report.get(author)
		perplexity_measure_file.write(author)
		perplexity_measure_file.write("|")
		for key in perplexity_measure.keys():
			perplexity = perplexity_measure.get(key)
			perplexity_measure_file.write(str(perplexity))
			perplexity_measure_file.write(" | ")
		perplexity_measure_file.write("\n")
		perplexity_measure_file.write("--------------------------------------------")
		perplexity_measure_file.write("\n")

	perplexity_measure_file.close()
	return 



# linear interpolation functions

def lamda_set_generator(n):
	lambda_set = []
	lamb = 0
	init = 1
	for i in range(n-1):
		lamb = random.uniform(0,init)
		while lamb <= 0:
			lamb = random.uniform(0,init)
		lambda_set.append(lamb)
		init = init - lamb
	last_lamb = 1
	for item in lambda_set:
		last_lamb = last_lamb - item
	lambda_set.append(last_lamb)
	
	return lambda_set
def calculate_linear_interpolation(n_gram,n,lambda_set,LM_list,smooth_method):
	prob = 0
	for i in range(n):
		word_list = n_gram[(n-1-i):]
		prob = lambda_set[i] * (1/nGram_for_perplexity(i+1,word_list,LM_list,smooth_method)) + prob
	prob = math.log(prob)

	return prob
# calculate the probability of the dev_set using linear interpolation with n gram, given a lambda ser, LM set and a dev set. 
def dev_set_prob_linear_interpolation(dev_set_union,n,lambda_set,LM_list,smooth_method):
	probability = 0
	for i in range(len(dev_set_union)):
		if i >= n - 1 :
			prob = calculate_linear_interpolation(dev_set_union[i-(n-1):i+1], n, lambda_set, LM_list,smooth_method)
			probability = prob + probability
	return probability
def get_best_lamda(dev_set,n,LM_list,smooth_method):
	
	dev_set_union = []
	for sent in dev_set:
		for word in sent:
			dev_set_union.append(word)
	best_lamda = None
	best_prob = float("-inf")
	for i in range(20):
		lambda_set = lamda_set_generator(n)
		probability = dev_set_prob_linear_interpolation(dev_set_union,n,lambda_set,LM_list,smooth_method)
		if probability > best_prob:
			best_prob = probability
			best_lamda = lambda_set
	return best_lamda	
def test_on_dev(author_LM_dict, dev_dict_for_all_authors, N_gram):
	smooth = "G_T"
	print_dict =  {"Austen":0,"Christie":0,"Dickens":0,"Doyle":0,"Thoureau":0,"Whitman":0,"Wilde":0}
	correctness_dict = {"Austen":None,"Christie":None,"Dickens":None,"Doyle":None,"Thoureau":None,"Whitman":None,"Wilde":None}
	for author in dev_dict_for_all_authors.keys():
		print(author)
		sentenceList = dev_dict_for_all_authors.get(author)
		full_count_dictionary = {"Austen":0,"Christie":0,"Dickens":0,"Doyle":0,"Thoureau":0,"Whitman":0,"Wilde":0}
		for sent in sentenceList:
			print(sent)
			optimal_prob = float("-inf")
			prediction = None
			for author_ in N_gram.keys():	
				prob = nGram_with_smoothing(N_gram.get(author_), sent, author_LM_dict.get(author_), smooth)
				
				if prob > optimal_prob:
					optimal_prob = prob
					prediction = author_
			
			full_count_dictionary[prediction] = full_count_dictionary.get(prediction) + 1
			if prediction == author:
				print_dict[author] =  print_dict.get(author) + 1
		correctness_dict[author] = full_count_dictionary
			
	return print_dict




# Good_Turing smooth probability calculation 

def smooth_method_prob(prob, count1, count2, smoothMethod, v, N_gram):
	k = 5
	if smoothMethod == "G_T":
		N_c_dict = get_N_c_dict(N_gram)
		if count1 > k:
			factor_1 = (count1 / count2)
			prob += math.log(factor_1)
		elif 1 <= count1 <= k :	
			N_c1 = N_c_dict[count1+1]
			N_c = N_c_dict[count1]	
			N_k1 = N_c_dict[k+1]
			N_k =  N_c_dict[k]
			N_1 = N_c_dict[1]
			a = (count1+1)*(N_c1/N_c)
			b = ((k+1)*count1*N_k1)/N_1
			d = 1 - (((k+1)*N_k1)/N_1)
			discount_factor = (a - b)/d
			prob += math.log((count1/count2)*discount_factor)
		elif count1 == 0:
			gained_distribution = (N_c_dict[1]/N_c_dict[0]) / N_c_dict[0]
			prob += math.log(gained_distribution)

	elif smoothMethod == "laplace":
		prob += math.log((count1 + 1)/ (count2 + v))
	else:
		if count1 != 0 and count2 != 0:
			prob += math.log(count1/count2)
		else:
			print("both denominator and numerator are zero.")

	return prob
def get_N_c_dict(N_gram):
	N_c_dict = {}
	total = 0
	for c in N_gram.values():
		if N_c_dict.get(c) == None:
			N_c_dict[c] = 1
		else:
			N_c_dict[c] = N_c_dict.get(c) + 1
		total = total + c
	N_c_dict[0] = total 

	return N_c_dict
def nGram_with_smoothing(n,word_list, LM_list ,smooth_method):	
	v = LM_list[5]
	
	Prob = 0
	if n == 1:
		total = 0
		n_Gram = LM_list[0]
		for value in n_Gram.values():
			total = total + value
		for i in range(len(word_list)):
			key = "["+"'"+word_list[i]+"'"+"]"
			if LM_list[0].get(key) == None:
				numerator_count = 0
			else:
				numerator_count = LM_list[0].get(key)
			Prob = smooth_method_prob(Prob,numerator_count,total,smooth_method,v,n_Gram)
	
	else:
		n_Gram = LM_list[n-1] 
		n_minus_1_Gram = LM_list[n-2]
		for i in range(len(word_list)):
			if i >= n-1:
				numerator = str(word_list[i-(n-1):i+1])
				denominator = str(word_list[i-(n-1):i])
				if numerator in n_Gram.keys():
					numerator_count = n_Gram.get(numerator)
				else:
					numerator_count = 0
				if denominator in n_minus_1_Gram.keys():	
					denominator_count = n_minus_1_Gram.get(denominator)
				else:
					denominator_count = 0
				Prob = smooth_method_prob(Prob, numerator_count, denominator_count, smooth_method, v, n_Gram)
			
	return Prob
def print_print_dict(print_dict,devSetLengthDict):
	for author in print_dict.keys():
		print(author)
		print("       ")
		print(print_dict.get(author))
		print("/" + str(devSetLengthDict.get(author)))
		print("\n")
def N_gram_count(stringList):
	#string = w1 w2 w3 w4 w4...
	unigram = {}
	bigram = {}
	trigram = {}
	quadgram = {}
	quintgram = {}
	vocabList = []
	for i in range(len(stringList)):
		if stringList[i] not in vocabList:
			vocabList.append(stringList[i])
		if unigram.get(str([stringList[i]])) == None:
			unigram[str([stringList[i]])] = 1
		else:
			unigram[str([stringList[i]])] = unigram[str([stringList[i]])] + 1
		if i > 0:
			if bigram.get(str(stringList[i-1:i+1])) == None:
				bigram[str(stringList[i-1:i+1])] = 1
			else:
				bigram[str(stringList[i-1:i+1])] = bigram[str(stringList[i-1:i+1])] + 1
		if i > 1:
			if trigram.get(str(stringList[i-2:i+1])) == None:
				trigram[str(stringList[i-2:i+1])] = 1
			else:
				trigram[str(stringList[i-2:i+1])] = trigram[str(stringList[i-2:i+1])] + 1
		if i > 2:
			if quadgram.get(str(stringList[i-3:i+1])) == None:
				quadgram[str(stringList[i-3:i+1])] = 1
			else:
				quadgram[str(stringList[i-3:i+1])] = quadgram[str(stringList[i-3:i+1])] + 1
		if i > 3:
			if quintgram.get(str(stringList[i-4:i+1])) == None:
				quintgram[str(stringList[i-4:i+1])] = 1
			else:
				quintgram[str(stringList[i-4:i+1])] = quintgram[str(stringList[i-4:i+1])] + 1
	return unigram, bigram, trigram, quadgram, quintgram, len(vocabList)
def clean_Data_And_Get_Dev_Training_Set(text):
	text_list = ["<s>"]
	line = re.sub(r'\\n+',r' ',text)
	devSet_size = 25
	devSet = []
	training = ["<s>"]
	i = 0
		
		# put whitespaces around unambiguous separators
	line = re.sub(r'([\\?!()\";/\\|])',r' \1 ',line)
		# put whitespaces around comma that are not inside numbers. 
	line = re.sub(r'([^0-9]),',r'\1 , ',line)
	line = re.sub(r',([^0-9])',r' , \1',line)

		# distinguish singlequotes from apostrophes by segmenting off single quotes not preceded by letter
	line = re.sub(r"^'",r" ' ",line)
	line = re.sub(r"([^a-zA-Z0-9])'",r"\1 ' ",line)

		# segment off unambiguous word-final punctuations.
	line = re.sub(r"('|:|-)",r" \1",line)
	line = re.sub(r"('|:|-)([^a-zA-Z])",r" \1 \2",line)

	one_sentence = ["<s>"]

	abbrevList = ['Mr.','Ms.','Mrs.','Doc.']

	# for every word in the book.
	for word in line.split(" "):
		if re.match(r".*\.",word):
			if re.match(r"[a-zA-Z0-9]+\.",word) and \
			not re.match(r'([a-zA-Z]\.[a-zA-Z]\.)+|[A-Z][bcdfghj-nptvxz]+\.',word) and word not in abbrevList:
				word = re.sub(r'\.',r' \.',word)
		if re.match(r'[?!]',word):
			if i < devSet_size: 
				one_sentence.append("</s>")
				if len(one_sentence) > 3:
					devSet.append(one_sentence)
					i = i + 1
				one_sentence = ["<s>"]	
		if re.match(r"^[a-zA-Z0-9]",word):
			chars = word.split(" ")		
			for char in chars:
				if re.match(r"[a-zA-Z0-9]+",char):
					text_list.append(char.lower())
					# also append to the dev list.
					if i < devSet_size:
						one_sentence.append(char.lower())
					else:
						training.append(char.lower())
				elif re.match(r"\\.",char):					
					text_list.append(char.lower())
					if i < devSet_size:
						one_sentence.append("</s>")
						if len(one_sentence) > 3:
							devSet.append(one_sentence)
							i = i + 1
						one_sentence = ["<s>"]						
					else:
						training.append("</s>")
						training.append("<s>")	

	training = training[:-1]

	return text_list,devSet,training
def get_author_dict(File):
	authors = open(File)
	author_dict = {}
	for authorString in authors:
		string = authorString.split(",")
		author = string[0]
		url = string[1]
		data = urllib.request.urlopen(url)
		text = str(data.read())		
		author_dict[author] = text
		
	return author_dict


# first test: trigram with laplace smoothing

if __name__ == "__main__":
	para = sys.argv[1]
	author_list = sys.argv[2]
	author_dict = get_author_dict(author_list)

	if para == "-dev":
		dev(author_dict)
	elif para == "-test":
		test_file = sys.argv[3]
		test(test_file,author_dict)
	else:
		print("wrong input")

