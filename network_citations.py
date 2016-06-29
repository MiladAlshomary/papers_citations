## Imports
from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.feature import Normalizer
import time 

## CONSTANTS
APP_NAME = "Network Citations"


def histogram():
	citations = sc.textFile("/corpora/corpus-microsoft-academic-graph/data//PaperReferences.tsv");
	papers_citations = citations.map(lambda l : (l.split('\t')[1], 1)).reduceByKey(lambda a,b: a+b)
	papers_citations.saveAsTextFile('/corpora/corpus-microsoft-academic-graph/data/PaperReferencesCounts.tsv')
	citations_stats = papers_citations.map(lambda c: c[1] )
	#histogram = citations_stats.histogram(list(range(0,200000,100)))
	#print(citations.stats())

def extract_cs_papers(sc):
	#field of study
	fos = sc.textFile("/corpora/corpus-microsoft-academic-graph/data/FieldsOfStudy.tsv.bz2")
	#onley computer science
	fos = fos.map(lambda x : x.split("\t")).filter(lambda x: "Computer Science" in x[1])
	#[['21286E67', 'Information and Computer Science'], ['0186E68F', 'AP Computer Science'], ['0271BC14', 'Computer Science'], ['08063736', 'On the Cruelty of Really Teaching Computer Science']]
	keywords = sc.textFile("/corpora/corpus-microsoft-academic-graph/data/PaperKeywords.tsv.bz2")
	#onley keywords related to computer science field
	fkws = keywords.map(lambda k: k.split("\t")).filter(lambda l: l[2] == "0271BC14")
	#onley papers that talks about computer science field with number of keywords as value
	cs_papers_ids = fkws.map(lambda kw: (kw[0], 1)).reduceByKey(lambda a,b : a+b)
	cs_papers_ids.saveAsHadoopFile('/user/bd-ss16-g3/data/cs_papers_ids', "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

def get_papers_of(sc, year):
	papers    = sc.textFile("/corpora/corpus-microsoft-academic-graph/data/Papers.tsv.bz2")
	papers = papers.map(lambda line : line.split("\t"))
	papers = papers.map(lambda line : [line[0], line[1], line[2], int(line[3]), line[4], line[5], line[6], line[7], line[8], line[9], line[10]]).filter(lambda l: l[3] == year)
	return papers

def get_citations_on_papers_of(sc, year):
	#papers and number of citations per year
	citations = sc.textFile("/corpora/corpus-microsoft-academic-graph/data/PaperReferences.tsv.bz2").map(lambda line : line.split("\t")).map(lambda c: (c[0], c[1]))

	papers = get_papers_of(sc, year)
	papers = papers.map(lambda p: (p[0], p[3]))


	#join
	papersMap = sc.broadcast(papers.collectAsMap());
	rowFunc1 = lambda x: (x[0], x[1], papersMap.value.get(x[1], -1))
	def mapFunc1(partition):
		for row in partition:
			yield rowFunc1(row)

	result = citations.mapPartitions(mapFunc1, preservesPartitioning=True)
	result = result.filter(lambda c: c[2] != -1).map(lambda x: (x[0], x[1]))
	return result

def nb_citations_per_paper(sc):
	papers  = sc.textFile("/user/bd-ss16-g3/data/papers")
	papers  = papers.map(lambda p : p.split("\t"))

	citations    = sc.textFile("/user/bd-ss16-g3/data/citations")
	cited_papers = citations.map(lambda x: x.split("\t")).map(lambda x: (x[1], 1)).reduceByKey(lambda a, b: a+b)

	#join
	cited_papers_bc = sc.broadcast(cited_papers.collectAsMap());
	rowFunc = lambda line: (line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8], line[9], line[10], cited_papers_bc.value.get(line[0], 0))
	def mapFunc(partition):
		for row in partition:
			yield rowFunc(row)

	result = papers.mapPartitions(mapFunc, preservesPartitioning=True)
	return result

def authors_weights(sc):
	papers    = sc.textFile("/user/bd-ss16-g3/data/papers_with_nb_citations")
	papers = papers.map(lambda p : p.split("\t"))
	#paper_id and number of citations
	papers = papers.map(lambda p: (p[0], p[11]))
	papers = papers.filter(lambda p: p[1] != '0')

	paa = sc.textFile("/corpora/corpus-microsoft-academic-graph/data/PaperAuthorAffiliations.tsv.bz2").map(lambda l : l.split("\t")).filter(lambda a : a[1] != '')
	#author_id & 1 for the paper he/she is in
	paa1 = paa.map(lambda l : (l[0], l[1], 1/float(l[5])));

	#join
	papersMap = sc.broadcast(papers.collectAsMap());

	rowFunc1 = lambda x: (x[1], float(papersMap.value.get(x[0], 0)) * x[2])
	def mapFunc1(partition):
		for row in partition:
			yield rowFunc1(row)


	stage1 = paa1.mapPartitions(mapFunc1, preservesPartitioning=True)
	
	stage1 = stage1.reduceByKey(lambda a, b : a+b)

	result = stage1.combineByKey(lambda value: (value, 1),lambda x, value: (x[0] + value, x[1] + 1),lambda x, y: (x[0] + y[0], x[1] + y[1]))
	result = result.map(lambda item: (item[0], item[1][0]/item[1][1]))

	return result

def affiliations_weights(sc):
	papers = sc.textFile("/user/bd-ss16-g3/data/papers_with_nb_citations")
	papers = papers.map(lambda p : p.split("\t"))
	#paper_id and number of citations
	papers = papers.map(lambda p: (p[0], p[11]))
	papers = papers.filter(lambda p: p[1] != '0')

	paa = sc.textFile("/corpora/corpus-microsoft-academic-graph/data/PaperAuthorAffiliations.tsv.bz2").map(lambda l : l.split("\t")).filter(lambda a : a[2] != '')
	#author_id & 1 for the paper he/she is in
	paa = paa.map(lambda l : (l[0], l[2], 1));

	#join
	papersMap = sc.broadcast(papers.collectAsMap());

	rowFunc1 = lambda x: (x[1], int(papersMap.value.get(x[0], 0)))
	def mapFunc1(partition):
		for row in partition:
			yield rowFunc1(row)


	result = paa.mapPartitions(mapFunc1, preservesPartitioning=True)
	result = result.reduceByKey(lambda a, b : a+b)

	return result

def conf_weights(sc):
	papers    = sc.textFile("/user/bd-ss16-g3/data/papers_with_nb_citations")
	papers = papers.map(lambda p : p.split("\t"))
	#conference_id and number of citations
	confs = papers.map(lambda p: (p[8].lower(), int(p[11])))
	confs = confs.filter(lambda p: p[1] != 0 and p[0] != '')

	confs = confs.reduceByKey(lambda a, b : a+b)

	return confs

def fos_weights(sc):
	papers    = sc.textFile("/user/bd-ss16-g3/data/papers_with_nb_citations")
	papers = papers.map(lambda p : p.split("\t"))
	#paper_id and number of citations
	papers = papers.map(lambda p: (p[0], p[11]))
	papers = papers.filter(lambda p: p[1] != '0')

	keywords = sc.textFile("/corpora/corpus-microsoft-academic-graph/data/PaperKeywords.tsv.bz2")
	keywords = keywords.map(lambda k: k.split("\t"))
	
	#join
	papersMap = sc.broadcast(papers.collectAsMap());

	rowFunc1 = lambda x: (x[2], int(papersMap.value.get(x[0], 0)))
	def mapFunc1(partition):
		for row in partition:
			yield rowFunc1(row)

	fos = keywords.mapPartitions(mapFunc1, preservesPartitioning=True)
	fos = fos.reduceByKey(lambda a, b : a+b)

	return fos

def get_paa_of(sc, year):
	#papers and number of citations per year
	paas = sc.textFile("/corpora/corpus-microsoft-academic-graph/data/PaperAuthorAffiliations.tsv.bz2")
	paas = paas.map(lambda line : line.split("\t")).map(lambda c: (c[0], c[1], c[2], c[3], c[4]))

	papers = get_papers_of(sc, year)
	papers = papers.map(lambda p: (p[0], 1))


	#join
	papersMap = sc.broadcast(papers.collectAsMap())
	rowFunc1 = lambda x: (x[0], x[1], x[2], x[3], x[4], papersMap.value.get(x[0], -1))
	def mapFunc1(partition):
		for row in partition:
			yield rowFunc1(row)

	result = paas.mapPartitions(mapFunc1, preservesPartitioning=True)
	result = result.filter(lambda c: c[5] != -1)
	return result

def convert_papers_to_feature_file(sc):
	#step1 conference weight
	conferences = sc.textFile("/user/bd-ss16-g3/data/confs_citations")
	conferences = conferences.map(lambda a : a.split("\t")).filter(lambda a: float(a[1]) > 0).map(lambda a: (a[0], a[1]))
	
	conferences_bc = sc.broadcast(conferences.collectAsMap())

	papers = sc.textFile("/user/bd-ss16-g3/data/papers").map(lambda l : l.split("\t"))
	#paper_id, conf_id
	papers = papers.map(lambda l : (l[0], l[8].lower()));

	rowFunc1  = lambda x: (x[0], float(conferences_bc.value.get(x[1], 0)))
	def mapFunc1(partition):
		for row in partition:
			yield rowFunc1(row)

	papers_with_conf_weights = papers.mapPartitions(mapFunc1, preservesPartitioning=True)
	#by now we have for each paper the weight of its authors
	papers_with_conf_weights.saveAsHadoopFile('/user/bd-ss16-g3/data/papers_conferences_weight', "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")


	#step2 adding author weight
	authors = sc.textFile("/user/bd-ss16-g3/data/authors_citations")
	authors = authors.map(lambda a : a.split("\t")).filter(lambda a: float(a[1]) > 0).map(lambda a: (a[0], a[1]))
	
	authors_bc = sc.broadcast(authors.collectAsMap())

	paa = sc.textFile("/user/bd-ss16-g3/data/paper_author_affiliation").map(lambda l : l.split("\t")).filter(lambda a : a[1] != '')
	#paper_id, author_id
	paa = paa.map(lambda l : (l[0], l[1]));

	rowFunc1  = lambda x: (x[0], float(authors_bc.value.get(x[1], 0)))
	def mapFunc1(partition):
		for row in partition:
			yield rowFunc1(row)

	papers_with_author_weights = paa.mapPartitions(mapFunc1, preservesPartitioning=True)
	papers_with_author_weights = papers_with_author_weights.combineByKey(lambda value: (value, 1),lambda x, value: (x[0] + value, x[1] + 1),lambda x, y: (x[0] + y[0], x[1] + y[1]))
	papers_with_author_weights = papers_with_author_weights.map(lambda item: (item[0], item[1][0]/item[1][1]))
	#by now we have for each paper the weight of its authors
	papers_with_author_weights.saveAsHadoopFile('/user/bd-ss16-g3/data/papers_authors_weight', "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")



	#step3 affiliation weight
	affiliations = sc.textFile("/user/bd-ss16-g3/data/affiliation_citations")
	affiliations = affiliations.map(lambda a : a.split("\t")).filter(lambda a: float(a[1]) > 0).map(lambda a: (a[0], a[1]))
	
	affiliations_bc = sc.broadcast(affiliations.collectAsMap())

	paa = sc.textFile("/user/bd-ss16-g3/data/paper_author_affiliation").map(lambda l : l.split("\t")).filter(lambda a : a[2] != '')
	#paper_id, affiliation_id
	paa = paa.map(lambda l : (l[0], l[2]));

	rowFunc1  = lambda x: (x[0], float(affiliations_bc.value.get(x[1], 0)))
	def mapFunc1(partition):
		for row in partition:
			yield rowFunc1(row)

	papers_with_affiliation_weights = paa.mapPartitions(mapFunc1, preservesPartitioning=True)
	papers_with_affiliation_weights = papers_with_affiliation_weights.combineByKey(lambda value: (value, 1),lambda x, value: (x[0] + value, x[1] + 1),lambda x, y: (x[0] + y[0], x[1] + y[1]))
	papers_with_affiliation_weights = papers_with_affiliation_weights.map(lambda item: (item[0], item[1][0]/item[1][1]))
	#by now we have for each paper the weight of its authors
	papers_with_affiliation_weights.saveAsHadoopFile('/user/bd-ss16-g3/data/papers_affiliation_weight', "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

	#step4 fieldofstudy weight
	fos = sc.textFile("/user/bd-ss16-g3/data/fos_citations")
	fos = fos.map(lambda a : a.split("\t")).filter(lambda a: float(a[1]) > 0).map(lambda a: (a[0], a[1]))
	
	fos_bc = sc.broadcast(fos.collectAsMap())

	keywords = sc.textFile("/corpora/corpus-microsoft-academic-graph/data/PaperKeywords.tsv.bz2").map(lambda l : l.split("\t"))
	#paper_id, field_of_study
	keywords = keywords.map(lambda l : (l[0], l[2]));

	rowFunc1  = lambda x: (x[0], float(fos_bc.value.get(x[1], 0)))
	def mapFunc1(partition):
		for row in partition:
			yield rowFunc1(row)

	papers_with_fos_weights = keywords.mapPartitions(mapFunc1, preservesPartitioning=True)
	papers_with_fos_weights = papers_with_fos_weights.combineByKey(lambda value: (value, 1),lambda x, value: (x[0] + value, x[1] + 1),lambda x, y: (x[0] + y[0], x[1] + y[1]))
	papers_with_fos_weights = papers_with_fos_weights.map(lambda item: (item[0], item[1][0]/item[1][1]))
	#by now we have for each paper the weight of its authors
	papers_with_fos_weights.saveAsHadoopFile('/user/bd-ss16-g3/data/papers_fosn_weight', "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")


def merge_features_files(sc):

	papers = sc.textFile("/user/bd-ss16-g3/data/papers_with_nb_citations").map(lambda line: line.split("\t")).map(lambda i: (i[0], float(i[11])))
	papers = papers.filter(lambda p: p[1] > 0).map(lambda p: (p[0],p[1]))


	authors = sc.textFile("/user/bd-ss16-g3/data/papers_authors_weight").map(lambda line: line.split("\t")).map(lambda i: (i[0], i[1]))
	affiliations = sc.textFile("/user/bd-ss16-g3/data/papers_affiliation_weight").map(lambda line: line.split("\t")).map(lambda i: (i[0], i[1]))
	fos = sc.textFile("/user/bd-ss16-g3/data/papers_fosn_weight").map(lambda line: line.split("\t")).map(lambda i: (i[0], i[1]))
	conferences = sc.textFile("/user/bd-ss16-g3/data/papers_conferences_weight").map(lambda line: line.split("\t")).map(lambda i: (i[0], i[1]))

	stage1 = papers.leftOuterJoin(authors)
	stage2 = stage1.leftOuterJoin(affiliations)
	stage3 = stage2.leftOuterJoin(conferences)
	stage4 = stage3.leftOuterJoin(fos)

	#('808B8671', ((((3.0, '2.25'), '19267.0'), '32.0'), '13052.0'))
	#paper_id, nb_citations, fos, conf, aff, author
	stage4 = stage4.map(lambda p: (p[0],p[1][1], p[1][0][1], p[1][0][0][1], p[1][0][0][0][1], p[1][0][0][0][0]))
	stage4 = stage4.map(lambda p: (p[0], '\t'.join([str(p[1]),str(p[2]), str(p[3]), str(p[4]), str(p[5])])))
	stage4.saveAsHadoopFile('/user/bd-ss16-g3/data/features_file', "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")


def extract_features(sc, year):
	#step1 extract papers of year year
	papers = get_papers_of(sc, year)
	papers = papers.map(lambda line: (line[0], '\t'.join([line[0], line[1], line[2], str(line[3]), line[4], line[5], line[6], line[7], line[8], line[9], line[10]])))
	papers.saveAsHadoopFile('/user/bd-ss16-g3/data/papers', "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")
        
	#step2 extract citations on papers of year year
	citations = get_citations_on_papers_of(sc, year)
	citations.saveAsHadoopFile('/user/bd-ss16-g3/data/citations', "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

	#step3 extract paper author affiliation
	paas = get_paa_of(sc, year)
	paas = paas.map(lambda line: (line[0], '\t'.join([line[1], line[2], line[3], line[4]])))
	paas.saveAsHadoopFile('/user/bd-ss16-g3/data/paper_author_affiliation', "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")
	
	#step4 extract number of citations for each paper in the subset
	cited_papers = nb_citations_per_paper(sc)
	cited_papers = cited_papers.map(lambda p: (p[0],'\t'.join([p[1], p[2], str(p[3]), p[4], p[5], p[6], p[7], p[8], p[9], p[10], str(p[11])])))
	cited_papers.saveAsHadoopFile('/user/bd-ss16-g3/data/papers_with_nb_citations', "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")
	
	#step5 give authors weight based on number of citations they got on each paper
	author_feature = authors_weights(sc)
	author_feature.saveAsHadoopFile('/user/bd-ss16-g3/data/authors_citations', "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")
	
	#step6 give affiliations weight based on number of citations 
	affiliation_feature = affiliations_weights(sc)
	affiliation_feature.saveAsHadoopFile('/user/bd-ss16-g3/data/affiliation_citations', "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")
	
	#step7 give conferences weight based on number of citations
	confs = conf_weights(sc)
	confs.saveAsHadoopFile('/user/bd-ss16-g3/data/confs_citations', "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")
	
	#step8 give fos weight based on number of citations
	fos = fos_weights(sc)
	fos.saveAsHadoopFile('/user/bd-ss16-g3/data/fos_citations', "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

def learn_model(sc, file_path, normalize):
	feature_file = sc.textFile(file_path).map(lambda l:l.split("\t"))

	points = feature_file.map(lambda f: LabeledPoint(f[1], f[2:]))
	
	#normalizing
	if normalize:
		nor      = Normalizer()
		labels   = points.map(lambda x: x.label)
		features = points.map(lambda x: x.features)
		points = labels.zip(nor.transform(features))
		points = points.map(lambda i: LabeledPoint(i[0], i[1]))

	training, testing = points.randomSplit([0.7,0.3],11)
	index = 0
	iterations = 100
	p_mse = -1
	converge = False
	result = {}
	while(not converge):
		x = time.clock()
		model = LinearRegressionWithSGD.train(training, iterations=iterations, step=0.00001,intercept=True,regType="l1")
		y = time.clock()
		print("========== time = " + str(y - x))
		preds = testing.map(lambda p: (p.label, model.predict(p.features)))
		MSE = preds.map(lambda r: (r[1] - r[0])**2).reduce(lambda x, y: x + y) / preds.count()
		print("========== MSE = " + str(MSE))
		if p_mse == MSE:
			converge = True

		iterations = iterations +100
		result[iterations] = MSE
		p_mse = MSE
	
	print(result)
	return model

def test(sc):
	#papers_c  = sc.textFile("/user/bd-ss16-g3/data_all/papers_citations_less_200c").map(lambda line : line.split("\t")).map(lambda c: (c[0], c[1]))
	#papers_i    = sc.textFile("/corpora/corpus-microsoft-academic-graph/data/Papers.tsv.bz2").map(lambda line : line.split("\t")).map(lambda c: (c[0], c[3]))
	#result = papers_c.join(papers_i)
	#result = result.map(lambda r: (r[0], '\t'.join([r[1][0], r[1][1]])))
	#result.saveAsHadoopFile("/user/bd-ss16-g3/data_all/papers_citations_less_200c_year", "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

	# citations = sc.textFile("/corpora/corpus-microsoft-academic-graph/data/PaperReferences.tsv.bz2").map(lambda line : line.split("\t")).map(lambda c: (c[0], c[1]))
	# papers = sc.textFile("/user/bd-ss16-g3/data_all/papers_citations_less_200c_year").map(lambda line : line.split("\t")).map(lambda c: (c[0], c[2]))
	# result = papers.join(citations)
	# result = result.map(lambda r: (r[0], '\t'.join([r[1][0], r[1][1]])))
	# result.saveAsHadoopFile("/user/bd-ss16-g3/data_all/citations_1", "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

	# citations_1 = sc.textFile("/user/bd-ss16-g3/data_all/citations_1").map(lambda line : line.split("\t")).map(lambda c: (c[2], (c[0], c[1])))
	# papers = sc.textFile("/user/bd-ss16-g3/data_all/papers_citations_less_200c_year").map(lambda line : line.split("\t")).map(lambda c: (c[0], c[2]))
	# result = papers.join(citations_1)
	# result = result.map(lambda r: (r[0], '\t'.join([r[1][0], r[1][1][0], r[1][1][1]])))
	# result.saveAsHadoopFile("/user/bd-ss16-g3/data_all/citations_2", "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

	# citations_2 = sc.textFile("/user/bd-ss16-g3/data_all/citations_2").map(lambda line : line.split("\t"))
	# c3years = citations_2.filter(lambda c: (int(c[3]) - int(c[1])) == 3).map(lambda c: (c[0], c[2]))
	# c3years.saveAsHadoopFile("/user/bd-ss16-g3/data_all/citations_3years_old", "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

	# citations3years = sc.textFile("/user/bd-ss16-g3/data_all/citations_3years_old").map(lambda line : line.split("\t"))
	# cited_papers = citations3years.map(lambda l: (l[1], 1)).reduceByKey(lambda a,b: a+b)
	# all_papers   = sc.textFile("/user/bd-ss16-g3/data_all/papers_citations_less_200c").map(lambda line : line.split("\t")).map(lambda c: (c[0], c[1]))
	# all_papers   = all_papers.leftOuterJoin(cited_papers)
	# #update all papers nb citations from cited_papers
	# all_papers = all_papers.map(lambda p: (p[0], p[1][1])).map(lambda p: (p[0], 0 if p[1] == None else p[1]))
	# all_papers.saveAsHadoopFile("/user/bd-ss16-g3/data_all/papers_citations_less_200c_3years_citations", "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

	#author feature
	# all_papers = sc.textFile("/user/bd-ss16-g3/data_all/papers_citations_less_200c_3years_citations").map(lambda p: p.split("\t")).map(lambda p: (p[0], float(p[1])))
	# paa = sc.textFile("/corpora/corpus-microsoft-academic-graph/data/PaperAuthorAffiliations.tsv.bz2").map(lambda l : l.split("\t")).filter(lambda a : a[1] != '')
	# paa = paa.map(lambda p: (p[0], (p[1], 1/float(p[5]))))
	# result = paa.join(all_papers)
	# result = result.map(lambda i: (i[1][0][0], 0 if i[1][1] == None else (i[1][0][1] * i[1][1]) ))
	# #reduce by combining
	# result = result.combineByKey(lambda value: (value, 1),lambda x, value: (x[0] + value, x[1] + 1),lambda x, y: (x[0] + y[0], x[1] + y[1]))
	# result = result.map(lambda item: (item[0], item[1][0]/item[1][1]))
	# result.saveAsHadoopFile("/user/bd-ss16-g3/data_all/authors_weights", "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

	#affiliation feature
	# all_papers = sc.textFile("/user/bd-ss16-g3/data_all/papers_citations_less_200c_3years_citations").map(lambda p: p.split("\t")).map(lambda p: (p[0], float(p[1])))
	# paa = sc.textFile("/corpora/corpus-microsoft-academic-graph/data/PaperAuthorAffiliations.tsv.bz2").map(lambda l : l.split("\t")).filter(lambda a : a[2] != '')
	# paa = paa.map(lambda p: (p[0], (p[2], 1/float(p[5]))))
	# result = paa.join(all_papers)
	# result = result.map(lambda i: (i[1][0][0], 0 if i[1][1] == None else (i[1][0][1] * i[1][1]) ))
	# #reduce by combining
	# result = result.combineByKey(lambda value: (value, 1),lambda x, value: (x[0] + value, x[1] + 1),lambda x, y: (x[0] + y[0], x[1] + y[1]))
	# result = result.map(lambda item: (item[0], item[1][0]/item[1][1]))
	# result.saveAsHadoopFile("/user/bd-ss16-g3/data_all/affiliations_weights", "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

	#fos feature
	# all_papers = sc.textFile("/user/bd-ss16-g3/data_all/papers_citations_less_200c_3years_citations").map(lambda p: p.split("\t")).map(lambda p: (p[0], float(p[1])))
	
	# keywords = sc.textFile("/corpora/corpus-microsoft-academic-graph/data/PaperKeywords.tsv.bz2").map(lambda k: k.split("\t")).map(lambda f: (f[0], f[2]))
	# result   = keywords.join(all_papers)
	# result   = result.map(lambda i: (i[1][0], 0 if i[1][1] == None else i[1][1]))
	# #reduce by combining
	# result = result.combineByKey(lambda value: (value, 1),lambda x, value: (x[0] + value, x[1] + 1),lambda x, y: (x[0] + y[0], x[1] + y[1]))
	# result = result.map(lambda item: (item[0], item[1][0]/item[1][1]))
	# result.saveAsHadoopFile("/user/bd-ss16-g3/data_all/fos_weights", "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")


	#conf feature
	# papers_citations = sc.textFile("/user/bd-ss16-g3/data_all/papers_citations_less_200c_3years_citations").map(lambda p: p.split("\t")).map(lambda p: (p[0], float(p[1])))
	# papers = sc.textFile("/corpora/corpus-microsoft-academic-graph/data/Papers.tsv.bz2").map(lambda p: p.split("\t")).map(lambda p: (p[0], p[5]))

	# result   = papers_citations.join(papers)
	# result   = result.map(lambda i: (i[1][1], float(i[1][0])))
	
	# #reduce by combining
	# result = result.combineByKey(lambda value: (value, 1),lambda x, value: (x[0] + value, x[1] + 1),lambda x, y: (x[0] + y[0], x[1] + y[1]))
	# result = result.map(lambda item: (item[0], item[1][0]/item[1][1]))
	# result.saveAsHadoopFile("/user/bd-ss16-g3/data_all/confs_weights", "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

	#Learning ============= papers + authors ================
	# paa = sc.textFile("/corpora/corpus-microsoft-academic-graph/data/PaperAuthorAffiliations.tsv.bz2").map(lambda l : l.split("\t")).filter(lambda a : a[1] != '')
	# paa = paa.map(lambda p: (p[1], p[0]))

	# #join with authors
	# authors_f = sc.textFile("/user/bd-ss16-g3/data_all/authors_weights").map(lambda a: a.split("\t")).map(lambda a: (a[0], float(a[1])))
	# result = paa.join(authors_f).map(lambda p: (p[1][0], 0 if p[1][1] == None else p[1][1]))
	# #sum up weights 
	# result = result.reduceByKey(lambda a,b: a+b)
	# #join with papers
	# papers = sc.textFile("/user/bd-ss16-g3/data_all/papers_citations_less_200c_year").map(lambda a: a.split("\t")).map(lambda a: (a[0], float(a[1])))
	# result2 = papers.join(result).map(lambda p: (p[0], p[1][0], 0 if p[1][1] == None else p[1][1]))
	# result2 = result2.map(lambda x: (x[0], '\t'.join([str(x[1]), str(x[2])])))
	# result2.saveAsHadoopFile("/user/bd-ss16-g3/data_all/paper_author_weight_citations", "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

	#Learning ============= papers + affiliations ================
	# paa = sc.textFile("/corpora/corpus-microsoft-academic-graph/data/PaperAuthorAffiliations.tsv.bz2").map(lambda l : l.split("\t")).filter(lambda a : a[2] != '')
	# paa = paa.map(lambda p: (p[2], p[0]))

	# #join with authors
	# affs_f = sc.textFile("/user/bd-ss16-g3/data_all/affiliations_weights").map(lambda a: a.split("\t")).map(lambda a: (a[0], float(a[1])))
	# result = paa.join(affs_f).map(lambda p: (p[1][0], 0 if p[1][1] == None else p[1][1]))
	# #sum up weights 
	# result = result.reduceByKey(lambda a,b: a+b)
	# #join with papers
	# papers = sc.textFile("/user/bd-ss16-g3/data_all/papers_citations_less_200c_year").map(lambda a: a.split("\t")).map(lambda a: (a[0], float(a[1])))
	# result2 = papers.join(result).map(lambda p: (p[0], p[1][0], 0 if p[1][1] == None else p[1][1]))
	# result2 = result2.map(lambda x: (x[0], '\t'.join([str(x[1]), str(x[2])])))
	# result2.saveAsHadoopFile("/user/bd-ss16-g3/data_all/paper_affiliations_weight_citations", "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

	#Learning ============= papers + fos ================
	# fos_papers = sc.textFile("/corpora/corpus-microsoft-academic-graph/data/PaperKeywords.tsv.bz2").map(lambda k: k.split("\t")).map(lambda f: (f[2], f[0]))
	# papers_citations = sc.textFile("/user/bd-ss16-g3/data_all/papers_citations_less_200c_year").map(lambda a: a.split("\t")).map(lambda a: (a[0], float(a[1])))
	# fos_weights = sc.textFile("/user/bd-ss16-g3/data_all/fos_weights").map(lambda a: a.split("\t")).map(lambda a: (a[0], float(a[1])))

	# #join with authors
	# result = fos_papers.join(fos_weights).map(lambda p: (p[1][0], 0 if p[1][1] == None else p[1][1]))
	# #sum up weights 
	# result = result.reduceByKey(lambda a,b: a+b)
	# #join with papers
	# result2 = papers_citations.join(result).map(lambda p: (p[0], p[1][0], 0 if p[1][1] == None else p[1][1]))
	# result2 = result2.map(lambda x: (x[0], '\t'.join([str(x[1]), str(x[2])])))
	# result2.saveAsHadoopFile("/user/bd-ss16-g3/data_all/paper_fos_weight_citations", "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

	#Learning ============= papers + confs ================
	# papers_citations = sc.textFile("/user/bd-ss16-g3/data_all/papers_citations_less_200c_3years_citations").map(lambda p: p.split("\t")).map(lambda p: (p[0], float(p[1])))
	# confs_papers = sc.textFile("/corpora/corpus-microsoft-academic-graph/data/Papers.tsv.bz2").map(lambda p: p.split("\t")).map(lambda p: (p[5], p[0]))
	# confs_weights = sc.textFile("/user/bd-ss16-g3/data_all/confs_weights").map(lambda a: a.split("\t")).map(lambda a: (a[0], float(a[1])))
	# #join with authors
	# result = confs_papers.join(confs_weights).map(lambda p: (p[1][0], 0 if p[1][1] == None else p[1][1]))
	# #sum up weights 
	# result = result.reduceByKey(lambda a,b: a+b)
	# #join with papers
	# result2 = papers_citations.join(result).map(lambda p: (p[0], p[1][0], 0 if p[1][1] == None else p[1][1]))
	# result2 = result2.map(lambda x: (x[0], '\t'.join([str(x[1]), str(x[2])])))
	# result2.saveAsHadoopFile("/user/bd-ss16-g3/data_all/paper_conf_weight_citations", "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

	#merge all features together
	#papers_citations = sc.textFile("/user/bd-ss16-g3/data_all/papers_citations_less_200c_3years_citations").map(lambda p: p.split("\t")).map(lambda p: (p[0], float(p[1])))
	author_feature   = sc.textFile("/user/bd-ss16-g3/data_all/paper_author_weight_citations").map(lambda p: p.split("\t")).map(lambda p: (p[0], (float(p[1]), float(p[2]))))
	affiliation_feature   = sc.textFile("/user/bd-ss16-g3/data_all/paper_affiliations_weight_citations").map(lambda p: p.split("\t")).map(lambda p: (p[0], float(p[2])))
	fos_feature   = sc.textFile("/user/bd-ss16-g3/data_all/paper_fos_weight_citations").map(lambda p: p.split("\t")).map(lambda p: (p[0], float(p[2])))
	conf_feature  = sc.textFile("/user/bd-ss16-g3/data_all/paper_conf_weight_citations").map(lambda p: p.split("\t")).map(lambda p: (p[0], float(p[2])))
	
	result = author_feature.join(affiliation_feature)
	result.cache()
	result = result.join(fos_feature)
	result.cache()
	result = result.join(conf_feature)

	#paper_id, nb_citations, author_weight, affiliation_w, fos_weight, conf_weight
	result = result.map(lambda x: (x[0], str(x[1][0][0][0][0]), str(x[1][0][0][0][1]), str(x[1][0][0][1]), str(x[1][0][1]), str(x[1][1])))
	result = result.map(lambda x: (x[0], '\t'.join(x[1:])))
	result.saveAsHadoopFile("/user/bd-ss16-g3/data_all/paper_all_weights", "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

	print(result.take(1))


if __name__ == "__main__":
	# Configure OPTIONS
	conf = SparkConf().setAppName(APP_NAME)
	conf = conf.setMaster("yarn-client")
	conf = conf.set("spark.executor.memory", "25g").set("spark.driver.memory", "25g").set("spark.mesos.executor.memoryOverhead", "10000")
	sc   = SparkContext(conf=conf)

	#test(sc)
	# model = learn_model(sc, "/user/bd-ss16-g3/data_all/paper_author_weight_citations", False)
	# model.save(sc,'/user/bd-ss16-g3/data_all/author_model')

	model = learn_model(sc, "/user/bd-ss16-g3/data_all/paper_fos_weight_citations", False)
	model.save(sc,'/user/bd-ss16-g3/data_all/fos_model')

	#step1
	#Extract weights for the features
	#extract_features(sc, 2012)

	#step2
	#build feature file
	#convert_papers_to_feature_file(sc)

	#step3
	#merge featurs files into one
	#merge_features_files(sc)

	#step4
	#learn a linear model from the feature file
	#model = learn_model(sc)
	#model.save(sc,'/user/bd-ss16-g3/data/my_model')

