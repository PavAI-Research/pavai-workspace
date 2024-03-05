from genai_at_work import config, logutil
logger = logutil.logging.getLogger(__name__)

import crewai
import langchain.tools as langchain_tools 
from langchain_openai import ChatOpenAI
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
import requests
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from chromadb.utils import embedding_functions

from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.document import Document
import newspaper
from newspaper import fulltext
import traceback

## embedding for vectordb 
embedding_function = HuggingFaceEmbeddings(model_name=config.config["NEWSCREW_EMBEDDING"])
query_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.config["NEWSCREW_EMBEDDING"]) 
## 768
if config.config["NEWSCREW_LLM_PROVIDER"]=="ollama":
    #base_url="http://192.168.0.18:12345"
    #model="openhermes"
    llm = Ollama(model=config.config["OLLAMA_NEWSCREW_LLM_MODEL_NAME"],base_url=config.config["OLLAMA_NEWSCREW_LLM_BASE_URL"])
else:
    # LLAMACPP
    os.environ["OPENAI_API_KEY"] = "EMPTY"
    os.environ["OPENAI_API_BASE"] = config.config["OPENAI_NEWSCREW_LLM_BASE_URL"]
    os.environ["MODEL_NAME"] = config.config["OPENAI_NEWSCREW_LLM_MODEL_NAME"]
    llm = ChatOpenAI(model_name=config.config["OPENAI_NEWSCREW_LLM_MODEL_NAME"], temperature=0.7)
    ## embedding_function = OpenAIEmbeddings()

# Tool 1 : Save the news articles in a database
NEWS_SEARCH_DB=config.config["NEWS_SEARCH_DB"]
CREWAI_JOB_DB=config.config["CREWAI_JOB_DB"]

# Tool 1 : Search API for news articles on the web
search_tool = DuckDuckGoSearchRun()

# Tool 2 : Search for news from web site
class SearchNewsDB:

    @staticmethod
    def latest_yahoo_news(topic:str):
        import requests
        from bs4 import BeautifulSoup
        import pandas as pd
        # Perform news scraping from Yahoo and extract the result into Pandas dataframe. 
        news_data = []
        for page in (0,21,41):
            url = 'https://news.search.yahoo.com/search?q={}&b={}'.format(topic,page)
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            for news_item in soup.find_all('div', class_='NewsArticle'):
                news_title = news_item.find('h4').text
                news_source = news_item.find('span','s-source').text
                news_description = news_item.find('p','s-desc').text
                news_link = news_item.find('a').get('href')
                news_time = news_item.find('span', class_='fc-2nd').text
                # Perform basic clean text.
                news_time = news_time.replace('·', '').strip()
                news_title = news_title.replace('•', '').strip()
                news_data.append([news_title, news_source, news_description, news_link, news_time])
                print("*****"*20)
                print(news_description)

        news_data_df = pd.DataFrame(news_data, columns=['Title','Source','Description','Link','Time'])
        #print(news_data_df.head(10))
        return news_data,  news_data_df

    @staticmethod
    def latest_yahoo_news_content(topic:str):
        import requests
        from bs4 import BeautifulSoup
        import pandas as pd
        # Perform news scraping from Yahoo and extract the result into Pandas dataframe. 
        news_data = []
        for page in (0,21,41):
            url = 'https://news.search.yahoo.com/search?q={}&b={}'.format(topic,page)
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            for news_item in soup.find_all('div', class_='NewsArticle'):
                news_title = news_item.find('h4').text
                news_source = news_item.find('span','s-source').text
                news_description = news_item.find('p','s-desc').text
                news_link = news_item.find('a').get('href')
                news_time = news_item.find('span', class_='fc-2nd').text
                # Perform basic clean text.
                news_time = news_time.replace('·', '').strip()
                news_title = news_title.replace('•', '').strip()
                news_data.append([news_title, news_source, news_description, news_link, news_time])
                print("*****"*20)
                print(news_description)

        news_data_df = pd.DataFrame(news_data, columns=['Title','Source','Description','Link','Time'])
        webpages="## Result Pages"
        for link in news_data_df["Link"]:
            try:
                pagedata = SearchNewsDB.scrap_website(link)
                webpages=webpages+"<p>".join(pagedata)+"</p><hr/>\n"
                webpages=webpages+"******************\n"                
            except Exception as e:
                print(e)
        #print(news_data_df.head(10))
        return webpages,  news_data_df

    @staticmethod
    def scrap_website(url:str):
        from newspaper import Article     
        try:         
            article = Article(url)
            article.title
            article.download()
            article.parse()
            article.nlp()
            article_text=f"published on {article.publish_date} by author {article.authors}\n Content: {article.text}"
            article_text=article_text+f"\n keywords: {article.keywords}\n"
            #article_text=article_text+f"\n\n summary: {article.summary}\n"
            return article.title, article.summary, article_text
        except Exception as e:
            print(e)
        return "", "", ""            

    @staticmethod
    def split_text(longtext:str):
        text_splitter = CharacterTextSplitter(chunk_size=384, chunk_overlap=100)
        docs = [Document(page_content=x) for x in text_splitter.split_text(longtext)]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=768, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)
        return splits

    @staticmethod
    def save_text(docsplits:list):
        from langchain_community.vectorstores import Chroma
        vectorstore = Chroma.from_documents(docsplits, embedding=embedding_function, persist_directory=NEWS_SEARCH_DB)
        print("saved text")

    @staticmethod
    def query_db(query:str,col_name:str="langchain",max_results:int=10, persist_directory=NEWS_SEARCH_DB):
        import chromadb
        client = chromadb.PersistentClient(path=persist_directory)
        col = client.get_collection(col_name,embedding_function=query_embedding_function)
        print("document in collection ", col.count())
        query_result = col.query(query_texts=query, n_results=max_results)
        return query_result

    @staticmethod
    def prepare_db(persist_directory=NEWS_SEARCH_DB):
        import chromadb
        client = chromadb.PersistentClient(path=persist_directory)
        for col in client.list_collections():
            client.delete_collection(col.name)
            client.get_or_create_collection(col.name)
            print("db collection clean up done!")                        
        return client

    @staticmethod
    def websearch(keywords:str,max_results=10,timelimit=None,region=None, backend="lite",safesearch="moderate",tool:str="duckduckgo"):
        from duckduckgo_search import DDGS
        results = []        
        with DDGS() as ddgs:
            ddgs_results = [r for r in ddgs.text(keywords=keywords,
                                                    safesearch=safesearch,
                                                    backend=backend,
                                                    region=region,
                                                    timelimit=timelimit,
                                                    max_results=max_results)]
            for r in ddgs_results:
                results.append(r)
        ## aggregate results
        sresults=[]
        for result in results:
            print("***"*20)
            print(result["title"])
            print(result["body"])    
            print(result["href"])    
            sresults.append([result["title"], result["body"], result["href"]])
        result_df = pd.DataFrame(sresults, columns=['Title','Description','Link'])
        return results, result_df

    @langchain_tools.tool("week old news ")
    def week_old_news(keywords:str,max_results=10,timelimit="w",region=None, backend="lite",safesearch="moderate",tool:str="duckduckgo"):
        """Fetch one week old news articles and process their contents."""        
        from duckduckgo_search import DDGS
        results = []        
        with DDGS() as ddgs:
            ddgs_results = [r for r in ddgs.text(keywords=keywords,
                                                    safesearch=safesearch,
                                                    backend=backend,
                                                    region=region,
                                                    timelimit=timelimit,
                                                    max_results=max_results)]
            for r in ddgs_results:
                results.append(r)
        ## aggregate results
        sresults=[]
        for result in results:
            print("***"*20)
            print(result["title"])
            print(result["body"])    
            print(result["href"])    
            sresults.append([result["title"], result["body"], result["href"]])
        result_df = pd.DataFrame(sresults, columns=['Title','Description','Link'])
        return results, result_df

    @langchain_tools.tool("month old news ")
    def month_old_news(keywords:str,max_results=10,timelimit="m",region=None, backend="lite",safesearch="moderate",tool:str="duckduckgo"):
        """Fetch one month old news articles and process their contents."""        
        from duckduckgo_search import DDGS
        results = []        
        with DDGS() as ddgs:
            ddgs_results = [r for r in ddgs.text(keywords=keywords,
                                                    safesearch=safesearch,
                                                    backend=backend,
                                                    region=region,
                                                    timelimit=timelimit,
                                                    max_results=max_results)]
            for r in ddgs_results:
                results.append(r)
        ## aggregate results
        sresults=[]
        for result in results:
            print("***"*20)
            print(result["title"])
            print(result["body"])    
            print(result["href"])    
            sresults.append([result["title"], result["body"], result["href"]])
        result_df = pd.DataFrame(sresults, columns=['Title','Description','Link'])
        return results, result_df

    @langchain_tools.tool("yearl old news ")
    def year_old_news(keywords:str,max_results=10,timelimit="y",region=None, backend="lite",safesearch="moderate",tool:str="duckduckgo"):
        """Fetch one year old news articles and process their contents."""        
        from duckduckgo_search import DDGS
        results = []        
        with DDGS() as ddgs:
            ddgs_results = [r for r in ddgs.text(keywords=keywords,
                                                    safesearch=safesearch,
                                                    backend=backend,
                                                    region=region,
                                                    timelimit=timelimit,
                                                    max_results=max_results)]
            for r in ddgs_results:
                results.append(r)
        ## aggregate results
        sresults=[]
        for result in results:
            print("***"*20)
            print(result["title"])
            print(result["body"])    
            print(result["href"])    
            sresults.append([result["title"], result["body"], result["href"]])
        result_df = pd.DataFrame(sresults, columns=['Title','Description','Link'])
        return results, result_df

    @langchain_tools.tool("Yahoo latest news search Tool")
    def yahoo_news(topic:str):
        """get latest news from yahoo and process it's contents."""        
        from newspaper import Article            
        news_data,  news_data_df = SearchNewsDB.latest_yahoo_news(topic)
        all_splits = []
        for article in news_data:
            #print(article[0],":", article[3])
            article.title, article.summary, article_text = SearchNewsDB.scrap_website(article[3])            
            print("### ",article_text)            
            if len(article.text.strip())>0:
                splits = SearchNewsDB.split_text(longtext=article_text)
                all_splits.extend(splits)  
        # Index the accumulated content splits if there are any
        if all_splits:
            retriever = SearchNewsDB.save_text(all_splits,topic)
            return retriever
        else:
            return "No content available for processing."

    @langchain_tools.tool("Web site news scrapting Tool")
    def newswebsite(self,query: str, news_urls:list=[], save_result:bool=True):
        """Scrap web site news and contents."""
        from langchain_community.vectorstores import Chroma
        all_splits = []
        for weburl in news_urls: 
            try:                
                loader = WebBaseLoader(weburl)
                docs = loader.load()
                print(docs)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                all_splits.extend(splits)  
            except Exception as e:
                print(e)
        # print(all_splits)
        # Index the accumulated content splits if there are any
        if all_splits:
            if save_result:
                vectorstore = Chroma.from_documents(all_splits, embedding=embedding_function, persist_directory=NEWS_SEARCH_DB)
                retriever = vectorstore.similarity_search(query)
                return retriever
            else:
                return all_splits
        else:
            return "No content available for processing."

    @langchain_tools.tool("News DB Tool")
    def newsapi(query: str, api_key:str):
        """Fetch news articles and process their contents."""
        from langchain_community.vectorstores import Chroma
        API_KEY = os.getenv(
            'NEWSAPI_KEY')  # Fetch API key from environment variable
        base_url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'sortBy': 'publishedAt',
            'apiKey': api_key,
            'language': 'en',
            'pageSize': 5,
        }

        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            return "Failed to retrieve news."

        articles = response.json().get('articles', [])
        all_splits = []
        for article in articles:
            # Assuming WebBaseLoader can handle a list of URLs
            loader = WebBaseLoader(article['url'])
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            all_splits.extend(splits)  # Accumulate splits from all articles

        # Index the accumulated content splits if there are any
        if all_splits:
            vectorstore = Chroma.from_documents(
                all_splits, embedding=embedding_function, 
                persist_directory=NEWS_SEARCH_DB)
            retriever = vectorstore.similarity_search(query)
            return retriever
        else:
            return "No content available for processing."

    @staticmethod
    def preparenews(query:str, col_name:str="langchain", max_results:int=10,timelimit:str="m", persist_directory=NEWS_SEARCH_DB):
        import chromadb
        # web search
        duckducksearch_available=True
        try:
            results, result_df = SearchNewsDB.websearch(keywords=query,max_results=max_results,timelimit=timelimit)
        except Exception as e:
            print(e)
            duckducksearch_available=False            
        #print(result_df.head(10))
        # yahoo news search
        news_data,  news_data_df = SearchNewsDB.latest_yahoo_news(topic=query)
        print(news_data_df.head(10))
        # merge result
        if duckducksearch_available:
            linksdf = pd.concat([result_df["Link"], news_data_df["Link"]],names=["links"],ignore_index=True)
        else:
            linksdf = news_data_df["Link"]     
        print(linksdf.head())
        SearchNewsDB.prepare_db()
        # ingest document
        for ind in linksdf.index:
            link = linksdf[ind]
            try:
                webpagedata = SearchNewsDB.scrap_website(link)
                splitdata = SearchNewsDB.split_text(" ".join(webpagedata))
                SearchNewsDB.save_text(docsplits=splitdata)
            except Exception as e:
                # bad links ignore or invalid data
                print(e)
                print(traceback.format_exc())
        # verify document saved         
        client = chromadb.PersistentClient(path=persist_directory)        
        col = client.get_collection(col_name,embedding_function=query_embedding_function)
        print("document in news collection ", col.count())         

    @staticmethod
    def searchreport(query:str,col_name:str="langchain", max_results:int=10,timelimit:str="m", persist_directory=NEWS_SEARCH_DB):
        import genai_at_work.summarizer as summarizer
        newsreport=""
        query_result = SearchNewsDB.query_db(query=query,max_results=max_results,persist_directory=NEWS_SEARCH_DB)
        for doc in query_result["documents"]:
            #print("FULL_TEXT\n", doc)
            newsreport = summarizer.GeneralTextSummarizer().summarize_text(doc)
            print("SUMMARZED TEXT\n",newsreport)
            break
        return newsreport

# Tool 3 : Get the news articles from local database
class GetNews:
    @langchain_tools.tool("Get News Tool")
    def news(query: str,persist_directory=CREWAI_JOB_DB) -> str:
        """Search Chroma DB for relevant news information based on a query."""
        from langchain_community.vectorstores import Chroma        
        vectorstore = Chroma(persist_directory=persist_directory,
                             embedding_function=embedding_function)
        retriever = vectorstore.similarity_search(query)
        return retriever

class NewsCrew():

    @staticmethod
    def run_search_job(query:str,persist_directory=CREWAI_JOB_DB)->str:

        SearchNewsDB.prepare_db(persist_directory)

        # 2. Creating Agents
        news_search_agent = crewai.Agent(
            role='News Seacher',
            goal='Generate key points for each news article from the latest news',
            backstory='Expert in analysing and generating key points from news content for quick updates.',
            tools=[search_tool, SearchNewsDB().yahoo_news, SearchNewsDB().month_old_news],
            allow_delegation=True,
            verbose=True,
            llm=llm
        )

        writer_agent = crewai.Agent(
            role='Writer',
            goal='Identify all the topics received. Use the Get News Tool to verify the each topic to search. Use the Search tool for detailed exploration of each topic. Summarise the retrieved information in depth for every topic.',
            backstory='Expert in crafting engaging narratives from complex information.',
            tools=[GetNews().news, search_tool],
            allow_delegation=True,
            verbose=True,
            llm=llm
        )

        # 3. Creating Tasks
        news_search_task = crewai.Task(
            description='Search for Epson POS Digitial Receipt Trend and create key points for each news.',
            agent=news_search_agent,
            tools=[SearchNewsDB().yahoo_news,search_tool]
        )

        writer_task = crewai.Task(
            description="""
            Go step by step.
            Step 1: Identify all the topics received.
            Step 2: Use the Get News Tool to verify the each topic by going through one by one.
            Step 3: Use the Search tool to search for information on each topic one by one. 
            Step 4: Go through every topic and write an in-depth summary of the information retrieved.
            Don't skip any topic.
            """,
            agent=writer_agent,
            context=[news_search_task],
            tools=[GetNews().news]
        )

        # 4. Creating Crew
        news_crew = crewai.Crew(
            agents=[news_search_agent, writer_agent],
            tasks=[news_search_task, writer_task],
            process=crewai.Process.sequential,
            manager_llm=llm
        )

        # 5. Execute the crew to see RAG in action
        result = news_crew.kickoff()
        return result


import urllib
import requests

def webquery(q):
    base_url = "https://api.duckduckgo.com/?q={}&format=json"
    resp = requests.get(base_url.format(urllib.parse.quote(q)))
    json = resp.json()
    return json

if __name__=="__main__":
    import pandas as pd

    query = "why should we use epson pos digital receipts"
    SearchNewsDB.preparenews(query=query)
    searchreport = SearchNewsDB.searchreport(query=query,max_results=10)

    result = webquery(query)
    print(result)

    #crewreport = NewsCrew.run_search_job(query)
    #print("***NewsCrew***\n",crewreport)    
    #print(crewreport)


