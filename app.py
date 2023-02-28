from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.sql.functions import monotonically_increasing_id, lit
from pyspark.sql.types import *
import pyspark.sql.functions as f
import streamlit as st
import streamlit.components.v1 as stc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cs
import matplotlib.pyplot as plt
import pandas as pd


def start1():
    UdemyDF = spark.read.csv('udemy_tech.csv', header=True, inferSchema=True)
    UdemyDF = UdemyDF.withColumn("Enrollment", UdemyDF.Enrollment.cast('int'))
    UdemyDF = UdemyDF.withColumn("Stars", UdemyDF.Stars.cast('double'))
    UdemyDF = UdemyDF.withColumn("Rating", UdemyDF.Rating.cast('int'))
    CourseraDF = spark.read.csv('Coursera.csv', header=True, inferSchema=True)
    CourseraDF = CourseraDF\
        .withColumn('Course Rating', CourseraDF['Course Rating'].cast('double'))\
        .withColumnRenamed('Course Name', 'Title')\
        .withColumnRenamed('Course Description', 'Summary')\
        .withColumnRenamed('Course Rating', 'Stars')\
        .withColumnRenamed('Course URL', 'Link')

    CourseraDF = CourseraDF.withColumn(
        "index", lit(9964) + monotonically_increasing_id())
    CourseraDF = CourseraDF.select([c for c in CourseraDF.columns if (
        c != 'University' and c != 'Difficulty Level' and c != 'Skills')])
    UdemyDF = UdemyDF.select(
        [c for c in UdemyDF.columns if (c != 'Enrollment' and c != 'Rating')])
    CourseraDF = CourseraDF.select(
        "index", "Title", "Summary", "Stars", "Link")

    CourseraDF = CourseraDF.withColumn("Source", lit('Coursera'))
    UdemyDF = UdemyDF.withColumn("Source", lit('Udemy'))
    DF = UdemyDF.union(CourseraDF)
    DF = DF.na.fill({'Title': 0.0, 'Summary': 0.0,
                    'Stars': 0.0, 'Link': 0.0}).na.fill('')
    df = DF

    columns = ['Title', 'Summary']
    minDFs = {'Title': 2.0, 'Summary': 4.0}
    preProcStages = []
    for col in columns:
        regexTokenizer = RegexTokenizer(
            gaps=False, pattern='\w+', inputCol=col, outputCol=col+'Token')
        stopWordsRemover = StopWordsRemover(
            inputCol=col+'Token', outputCol=col+'SWRemoved')
        countVectorizer = CountVectorizer(
            minDF=minDFs[col], inputCol=col+'SWRemoved', outputCol=col+'TF')
        idf = IDF(inputCol=col+'TF', outputCol=col+'IDF')
        preProcStages += [regexTokenizer,
                          stopWordsRemover, countVectorizer, idf]

    pipeline = Pipeline(stages=preProcStages)
    model = pipeline.fit(df)
    df = model.transform(df)
    df = df.select('index', 'TitleIDF', 'SummaryIDF',
                   'Stars', 'Link', 'Source')
    DF_pandas = DF
    DF_pandas = DF_pandas.toPandas()
    TDF = DF.toPandas()
    RDF = DF.toPandas()
    data_collect = df.collect()
    return DF_pandas, TDF, RDF, df, data_collect


def cosine_similarityy(X, Y):
    denom = X.norm(2) * Y.norm(2)
    if denom == 0.0:
        return -1.0
    else:
        return X.dot(Y) / float(denom)


def search(string, DF_pandas, num_of_rec):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(DF_pandas['Title'])
    string_vector = vectorizer.transform([string])
    cosine_sim = cs(string_vector, vectors)
    cos = []
    for i in range(len(DF_pandas['Title'])):
        cos.append(cosine_sim[0][i])
    DF_pandas['cosine_sim'] = cos
    DF_pandas = DF_pandas.sort_values(by=["cosine_sim"], ascending=False)
    return(DF_pandas.head(num_of_rec))


def Sort(sub_li):
    l = len(sub_li)
    for i in range(0, l):
        for j in range(0, l-i-1):
            if (sub_li[j][1] < sub_li[j + 1][1]):
                tempo = sub_li[j]
                sub_li[j] = sub_li[j + 1]
                sub_li[j + 1] = tempo
    return sub_li


def recomm(x, TDF, num_of_rec, df, data_collect):
    gProd1 = df.filter(df['index'] == x).collect()[0]
    l = []
    for row in data_collect:
        c = cosine_similarityy(row['TitleIDF'], gProd1['TitleIDF'])
        i = row['index']
        l += [c]
    tit = []

    for i in range(len(TDF['Title'])):
        tit.append(l[i])
    TDF['titlesim'] = tit
    TDF = TDF.sort_values(by=["titlesim"], ascending=False)
    TDF = TDF.iloc[1:, :]
    return(TDF.head(num_of_rec))


def recommend(x, RDF, num_of_rec, df, data_collect):
    gProd1 = df.filter(df['index'] == x).collect()[0]
    l = []
    for row in data_collect:
        c = cosine_similarityy(row['SummaryIDF'], gProd1['SummaryIDF'])
        i = row['index']
        l += [(c)]
    rec = []

    for i in range(len(RDF['Title'])):
        rec.append(l[i])
    RDF['sim'] = rec
    RDF = RDF.sort_values(by=["sim"], ascending=False)
    RDF = RDF.iloc[1:, :]
    return(RDF.head(num_of_rec))


def compare(x, y, TDF):
    dff = TDF.iloc[x].values
    df2 = TDF.iloc[y].values
    data = []
    for i in range(len(dff)):
        data.append(str(dff[i]))
    data.pop(0)
    cdf = pd.DataFrame(['Tile', 'Summary', 'Rating', 'Link', 'Source'])
    cdf['ind'] = ['Tile', 'Summary', 'Rating', 'Link', 'Source']
    data2 = []
    for i in range(len(df2)):
        data2.append(str(df2[i]))
    data2.pop(0)
    cdf[1] = data2
    cdf[2] = data
    return cdf


def toprated(c, RDF, df, data_collect, num_of_rec):
    def recommend1(x, RDF):
        gProd1 = df.filter(df['index'] == x).collect()[0]
        l = []
        for row in data_collect:
            c = cosine_similarityy(row['SummaryIDF'], gProd1['SummaryIDF'])
            i = row['index']
            l += [(c)]
        rec = []

        for i in range(len(RDF['Title'])):
            rec.append(l[i])
        RDF['sim'] = rec
        RDF = RDF.sort_values(by=["sim"], ascending=False)
        RDF = RDF.iloc[1:, :]

        return(RDF.head(num_of_rec))
    return(recommend1(c, RDF).sort_values(by=["Stars"], ascending=False))


RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:25px;
box-shadow:0 0 15px 5px #ccc; background-color: #a8f0c6;
  border-left: 5px solid #6c6c6c;">
<h2 style="color:purple;">{}</h2>
<p style="color:blue;"><span style="color:black;">📈Ratings ::  </span>{}</p>
<p style="color:blue;"><span style="color:black;">🔗Course-ID ::  </span>{}</p>
                        
<p style="color:blue;"><span style="color:black;">💲Source ::  </span>{}</p>
<p style="color:blue;"><span style="color:black;">🧑‍🎓👨🏽‍🎓 Summary ::  </span>{}</p>
</div>
"""


@st.cache
def search_term_if_not_found(term, df):
    result_df = df[df['course_title'].str.contains(term)]
    return result_df


def main():

    st.title("Course Recommendation App")

    menu = ["Home", "Recommendation based on Title",
            "Recommendation based on Summary", "Compare", "Top rated similar courses", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader(
            "Welcome!!")
        DF_pandas, TDF, RDF, df, data_collect = start1()
        search_term = st.text_input("Please enter the course name to search")
        num_of_rec = st.sidebar.number_input(
            "Select number of courses to display", 4, 30, 7)
        if st.button("Search"):
            if search_term is not None:
                try:
                    results = search(search_term, DF_pandas, num_of_rec)
                    with st.expander("Results as JSON"):
                        results_json = results.to_dict('index')
                        st.write(results_json)
                    for row in results.iterrows():
                        id = row[1][0]
                        title = row[1][1]
                        summary = row[1][2]
                        stars = row[1][3]
                        url = row[1][4]
                        source = row[1][5]

                        st.write(f'''
    <a target="_self" href="{url}"><button>LINK</button></a>''', unsafe_allow_html=True)
                        stc.html(RESULT_TEMP.format(title, stars,
                                 id, source, summary), height=350)

                except:
                    results = "Not Found"
                    st.warning(results)
                    st.info("Suggested Options include")
                    result_df = search_term_if_not_found(
                        search_term, DF_pandas)
                    st.dataframe(result_df)
        else:
            st.dataframe(DF_pandas)

    elif choice == "Recommendation based on Title":
        st.subheader("Recommendation based on Title")
        DF_pandas, TDF, RDF, df, data_collect = start1()
        search_term = st.text_input("Please enter the Course-ID")
        num_of_rec = st.sidebar.number_input(
            "Select number of courses to display", 4, 30, 7)
        if st.button("Recommend Courses"):
            if search_term is not None:
                try:
                    search_term = int(search_term)
                    results = recomm(search_term, TDF,
                                     num_of_rec, df, data_collect)
                    with st.expander("Results as JSON"):
                        results_json = results.to_dict('index')
                        st.write(results_json)

                    for row in results.iterrows():
                        id = row[1][0]
                        title = row[1][1]
                        summary = row[1][2]
                        stars = row[1][3]
                        url = row[1][4]
                        source = row[1][5]

                        st.write(f'''
    <a target="_self" href="{url}"><button>LINK</button></a>''', unsafe_allow_html=True)
                        stc.html(RESULT_TEMP.format(title, stars,
                                 id, source, summary), height=350)

                except:
                    results = "Not Found"
                    st.warning(results)
                    st.write(
                        "Please enter the Course-ID in number format(ex : 456)")

        else:
            st.write("")

    elif choice == "Recommendation based on Summary":
        st.subheader("Recommendation based on Summary")
        DF_pandas, TDF, RDF, df, data_collect = start1()
        search_term = st.text_input("Please enter the Course-ID")
        num_of_rec = st.sidebar.number_input(
            "Select number of courses to display", 4, 30, 7)
        if st.button("Recommend Courses"):
            if search_term is not None:
                try:
                    search_term = int(search_term)
                    results = recommend(
                        search_term, RDF, num_of_rec, df, data_collect)
                    with st.expander("Results as JSON"):
                        results_json = results.to_dict('index')
                        st.write(results_json)

                    for row in results.iterrows():
                        id = row[1][0]
                        title = row[1][1]
                        summary = row[1][2]
                        stars = row[1][3]
                        url = row[1][4]
                        source = row[1][5]

                        st.write(f'''
    <a target="_self" href="{url}"><button>LINK</button></a>''', unsafe_allow_html=True)
                        stc.html(RESULT_TEMP.format(title, stars,
                                 id, source, summary), height=350)

                except:
                    results = "Not Found"
                    st.warning(results)
                    st.write(
                        "Please enter the Course-ID in number format(ex : 456)")
        else:
            st.write("")

    elif choice == "Compare":
        st.subheader("Comparison between 2 courses")
        DF_pandas, TDF, RDF, df, data_collect = start1()
        search_term1 = st.text_input("Please enter the Course-ID 1")
        search_term2 = st.text_input("Please enter the Course-ID 2")
        if st.button("Compare"):
            if search_term1 and search_term2 is not None:
                try:
                    search_term1 = int(search_term1)
                    search_term2 = int(search_term2)
                    results = compare(
                        search_term1, search_term2, TDF).set_index('ind')
                    st.dataframe(results)

                except:
                    results = "Not Found"
                    st.warning(results)
                    st.write(
                        "Please enter the Course-ID in number format(ex : 456)")
        else:
            st.write("")

    elif choice == "Top rated similar courses":
        st.subheader("Top rated similar courses")
        DF_pandas, TDF, RDF, df, data_collect = start1()
        search_term1 = st.text_input("Please enter the Course-ID")
        num_of_rec = st.sidebar.number_input(
            "Select number of courses to display", 4, 30, 7)
        if st.button("Recommend"):
            if search_term1 is not None:
                try:
                    search_term1 = int(search_term1)
                    results = toprated(search_term1, RDF, df,
                                       data_collect, num_of_rec)
                    with st.expander("Results as JSON"):
                        results_json = results.to_dict('index')
                        st.write(results_json)

                    for row in results.iterrows():
                        id = row[1][0]
                        title = row[1][1]
                        summary = row[1][2]
                        stars = row[1][3]
                        url = row[1][4]
                        source = row[1][5]

                        st.write(f'''
    <a target="_self" href="{url}"><button>LINK</button></a>''', unsafe_allow_html=True)
                        stc.html(RESULT_TEMP.format(title, stars,
                                 id, source, summary), height=350)

                except:
                    results = "Not Found"
                    st.warning(results)
                    st.write(
                        "Please enter the Course-ID in number format(ex : 456)")
        else:
            st.write("")

    else:
        st.subheader("About")
        st.text("Built with Streamlit and Pandas by students of New Horizon of Engineering\n1. Charan C(1NH20AI020)\n2. Fizza Mirza(1NH20AI030)\n3. M Karthik Kumar(1NH20AI055)")


if __name__ == '__main__':
    spark = SparkSession.builder.master("local[1]").appName(
        "SparkByExamples.com").getOrCreate()
    main()
