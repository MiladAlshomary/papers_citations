citations = sc.textFile("/home/milad/University/Big data/LongTalk/citation-network/data/PaperReferences.tsv");


#each paper with number of citations
papers_citations = citations.map(lambda l : (l.split('\t')[1], 1)).reduceByKey(lambda a,b: a+b)
papers_citations.saveAsTextFile('/home/milad/University/Big data/LongTalk/citation-network/data/PaperReferencesCounts.tsv')

citations_stats = papers_citations.map(lambda c: c[1] )

#max = 178438
#histogram
histogram = citations_stats.histogram(list(range(0,200000,100)))


#papers and number of citations per year
papers    = sc.textFile("/corpora/corpus-microsoft-academic-graph/data/Papers.tsv.bz2");
citations = sc.textFile("/corpora/corpus-microsoft-academic-graph/data/PaperReferences.tsv.bz2");
citations = citations.map(lambda line : line.split("\t")).map(lambda c: (c[1], c))
papers    = papers.map(lambda line : line.split("\t")).map(lambda d : (d[0], d))

#crashed everything =D
paper_citations = papers.join(citations)
paper_citations.map(lambda line: )


#keywords
kwords = sc.textFile("/corpora/corpus-microsoft-academic-graph/data/PaperKeywords.tsv.bz2");
kwords = kwords.map(lambda l : l.split("\t"));
kwords = kwords.map(lambda l : (l[1], 1));
kwords = kwords.reduceByKey(lambda a,b: a+b);

topkwords = kwords.top(10, key=lambda x: x[1])

#[('humanidades', 720429), ('bioinformatics', 700032), ('biomedical research', 624874), ('kinetics', 449695), ('genetics', 438081), ('spectrum', 427706), ('biology', 358565), ('medicine', 356349), ('mathematical model', 338009), ('magnetic field', 335559)]


#authors
paa = sc.textFile("/corpora/corpus-microsoft-academic-graph/data/PaperAuthorAffiliations.tsv.bz2");
paa = paa.map(lambda l : l.split("\t"));
#filter only authors who are #1 or #2
paa = paa.filter(lambda a : a[1] != '' and (a[5] == '1' or a[5] == '2'));
paa = paa.map(lambda l : (l[1], 1));

paa = paa.reduceByKey(lambda a,b: a+b);

#to extract histogram
authors_1 = paa.map(lambda t: t[1])
#(count: 93063060, mean: 2.150490506114886, stdev: 25.2860897038, max: 153915.0, min: 1.0)
#[('816AFF85', 'united vertical media gmbh', 153915), ('122BD4EA', 'nurnberg', 153915), ('8042AED4', 'dmcgrath at hr com', 37505), ('80080CC4', 'fernando jose la calle prada', 37397), ('7F5B5DD6', 'alejandro sevilla fuentes', 37371), ('4B827820', 'iip digital', 16976), ('812FAAEC', 'bruno red kicinski', 14041), ('0FA867EC', 'ludwik adam wydaw dmuszewski', 14038), ('7F32ABD5', 'imago it gmbh', 12761), ('5A42B3AD', 'kaiserslautern', 12761)]

#for doing joins to know the names of these authors 
names   = sc.textFile("/corpora/corpus-microsoft-academic-graph/data/Authors.tsv.bz2")
names = names.map(lambda n : n.split("\t"))
paa = paa.filter(lambda a: a[1] > 50000);
paaMap = sc.broadcast(paa.collectAsMap());
rowFunc = lambda x: (x[0], x[1], paaMap.value.get(x[0], -1))
def mapFunc(partition):
    for row in partition:
        yield rowFunc(row)

result = names.mapPartitions(mapFunc, preservesPartitioning=True)


#field of study
fos = sc.textFile("/corpora/corpus-microsoft-academic-graph/data/FieldsOfStudy.tsv.bz2");
#onley computer science
fos = fos.map(lambda x : x.split("\t")).filter(lambda x: "Computer Science" in x[1]);
#[['21286E67', 'Information and Computer Science'], ['0186E68F', 'AP Computer Science'], ['0271BC14', 'Computer Science'], ['08063736', 'On the Cruelty of Really Teaching Computer Science']]
keywords = sc.textFile("/corpora/corpus-microsoft-academic-graph/data/PaperKeywords.tsv.bz2");
#onley keywords related to computer science field
fkws = keywords.map(lambda k: k.split("\t")).filter(lambda l: l[2] == "0271BC14");
#onley papers that talks about computer science field with number of keywords as value
cs_papers = fkws.map(lambda kw: (kw[0], 1)).reduceByKey(lambda a,b : a+b);
#save papers as hdfs
cs_papers.saveAsHadoopFile('/user/bd-ss16-g3/data/cs_papers', "org.apache.hadoop.mapred.TextOutputFormat", compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec");
