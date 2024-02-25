### Web Data Loader ###
# WebBaseLoader
# SeleniumURLLoader,
# NewsURLLoader

#!pip install langchain openai unstructured selenium newspaper3k textstat tiktoken faiss-cpu
# pip install selenium
# pip install unstructured
# pip install -U langchain-community
# pip install newspaper3k
# pip install selenium
# pip install webdriver-manager
# pip install duckduckgo-search==4.1.1

# import filedata
# from langchain.document_loaders import (WebBaseLoader, UnstructuredURLLoader, 
#                                         NewsURLLoader, SeleniumURLLoader)
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# https://newspaper.readthedocs.io/en/latest/
import os
from dotenv import dotenv_values
config = {
    **dotenv_values("env.shared"),  # load shared development variables
    **dotenv_values("env.secret"),  # load sensitive variables
    **os.environ,  # override loaded values with environment variables
}
import tiktoken

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-4-1106-preview")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def scrap_web_selenium(website_url:str, extract_element:str="/html/body"):
    import html2text  
    import selenium
    from selenium import webdriver
    from selenium.webdriver.common.by import By    

    print(selenium.__version__)
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    driver = webdriver.Chrome(options=options)
    text=None
    try:
        driver.get(website_url)
        title = driver.title
        driver.implicitly_wait(0.5)
        if extract_element:
            # Printing the whole body text ("/html/body")
            element_text = driver.find_element(By.XPATH,extract_element).text
            text=html2text.html2text(element_text)        
        else:
            html=driver.page_source
            text=html2text.html2text(html)
    except Exception as e:
        print("Exception ocurred ",e)
    finally:
        driver.quit()
    return text

## preserve original language
def scrap_web_newspaper(website_url, language:str=None):
    from newspaper import Article
    if language:
        a = Article(website_url, language=language)
    else:
        a = Article(website_url)         
    a.download()
    a.parse()
    a.nlp()
    content=f"{a.title}\n{a.text}\npublished date:{a.publish_date}\nauthors:{a.authors}"
    #print(a.summary)
    #print(a.keywords)        
    return content

def scrap_web(website_url:str, chatbot:list=[],history:list=[], tool:str="selenium"):
    import time
    t0=time.perf_counter()    

    if tool=="selenium":
        text_content= scrap_web_selenium(website_url=website_url, extract_element="/html/body")
    elif tool=="newspaper":
        text_content= scrap_web_newspaper(website_url)
    else:
        raise ValueError(f"Unsupported web scrapping tool {tool}")

    t1=time.perf_counter()            
    took=(t1-t0)    
    chatbot=[] if chatbot is None else chatbot  
    chatbot.append((f"load web page {website_url}", text_content))
    ## update history
    history=[] if history is None else history    
    history.append({"role": "user", "content": f"load web page {website_url}\n{text_content}"})        
    print(f"scrap_web took {took}s")
    return chatbot, history 

async def agent_results(word):
    from duckduckgo_search import AsyncDDGS
    results = []        
    async with AsyncDDGS() as ddgs:
        ddgs_results = [r async for r in ddgs.text(word, max_results=5)]
        for r in ddgs_results:
            results.append(r)
        return results

def web_search(keywords:str, chatbot:list=[],history:list=[],max_results=5,timelimit=None,region=None, backend="lite",safesearch="moderate",tool:str="duckduckgo"):
    import time
    import asyncio

    t0=time.perf_counter()    
    results = []
    results = asyncio.run(agent_results(keywords))
    formatted_output=""
    if isinstance(results,list):
        for r in results:
            formatted_output=formatted_output+"<b>"+r["title"]+"</b>\n"+r["href"]+"\n"+r["body"]+"<hr/>"
    #return formatted_output
    t1=time.perf_counter()            
    took=(t1-t0)    
    chatbot=[] if chatbot is None else chatbot  
    chatbot.append((f"web search for {keywords}", formatted_output))
    ## update history
    history=[] if history is None else history    
    history.append({"role": "user", "content": f"web search for {keywords}\nResult found:\n{formatted_output}"})        
    print(f"scrap_web took {took}s")
    return chatbot, history 

if __name__=="__main__":
    website = "https://phys.org/news/2023-11-qa-dont-blame-chatbots.html"
    #website = "http://www.bbc.co.uk/zhongwen/simp/chinese_news/2012/12/121210_hongkong_politics.shtml"    
    #website = "https://health.sina.com.cn/hc/2018-03-27/doc-ifysqfnh2148398.shtml"
    #result=scrap_web_newspaper(website)
    #result=scrap_web_selenium(website, extract_element="/html/body")
    #result = scrap_web(website,tool="newspaper")
    result = web_search(keywords="godzilla+2024")
    print(result)
    
    #loader_doc = load_document(WebBaseLoader, website)
    #loader_doc = load_document(SeleniumURLLoader, website)
    #loader_doc = load_document(NewsURLLoader, website)
    #print(loader_doc)

