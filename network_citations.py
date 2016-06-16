## Imports
from pyspark import SparkConf, SparkContext

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
	citations = sc.textFile("/corpora/corpus-microsoft-academic-graph/data/PaperReferences.tsv.bz2")
	citations = citations.map(lambda line : line.split("\t")).map(lambda c: (c[0], c[1]))

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
	confs = papers.map(lambda p: (p[10], int(p[11])))
	confs = confs.filter(lambda p: p[1] != 0)

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

def convert_papers_to_feature_file(sc):
	#step1 adding author weight
	authors = sc.textFile("/user/bd-ss16-g3/data/authors_citations")
	authors = authors.map(lambda a : a.split("\t")).filter(lambda a: float(a[1]) > 0).map(lambda a: (a[0], a[1]))
	
	authors_bc = sc.broadcast(authors.collectAsMap())

	paa = sc.textFile("/corpora/corpus-microsoft-academic-graph/data/PaperAuthorAffiliations.tsv.bz2").map(lambda l : l.split("\t")).filter(lambda a : a[1] != '')
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

	
def extract_features(sc, year):
	#step1 extract papers of year year
	papers = get_papers_of(year)
	papers = papers.map(lambda line: (line[0], '\t'.join([line[0], line[1], line[2], str(line[3]), line[4], line[5], line[6], line[7], line[8], line[9], line[10]])))
	papers.saveAsHadoopFile('/user/bd-ss16-g3/data/papers', "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

	#step2 extract citations on papers of year year
	citations = get_citations_on_papers_of(year)
	citations.saveAsHadoopFile('/user/bd-ss16-g3/data/citations', "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

	#step3 extract number of citations for each paper in the subset
	cited_papers = nb_citations_per_paper(sc)
	cited_papers = cited_papers.map(lambda p: (p[0],'\t'.join([p[1], p[2], str(p[3]), p[4], p[5], p[6], p[7], p[8], p[9], p[10], str(p[11])])))
	cited_papers.saveAsHadoopFile('/user/bd-ss16-g3/data/papers_with_nb_citations', "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

	#step4 give authors weight based on number of citations they got on each paper
	author_feature = authors_weights(sc)
	author_feature.saveAsHadoopFile('/user/bd-ss16-g3/data/authors_citations', "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

	#step5 give affiliations weight based on number of citations 
	affiliation_feature = affiliations_weights(sc)
	affiliation_feature.saveAsHadoopFile('/user/bd-ss16-g3/data/affiliation_citations', "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

	#step6 give conferences weight based on number of citations
	confs = conf_weights(sc)
	confs.saveAsHadoopFile('/user/bd-ss16-g3/data/confs_citations', "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

	#step7 give fos weight based on number of citations
	fos = fos_weights(sc)
	fos.saveAsHadoopFile('/user/bd-ss16-g3/data/fos_citations', "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")

if __name__ == "__main__":
	# Configure OPTIONS
	conf = SparkConf().setAppName(APP_NAME)
	conf = conf.setMaster("yarn-client")
	conf = conf.set("spark.executor.memory", "25g").set("spark.driver.memory", "25g").set("spark.mesos.executor.memoryOverhead", "10000")
	sc   = SparkContext(conf=conf)

	#step1
	#Extract weights for the features
	#extract_features(sc, 2012)

	#step2
	#build feature file
	convert_papers_to_feature_file(sc)