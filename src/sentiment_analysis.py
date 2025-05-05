from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate


def analyze_sentiment(query: str) -> str:

    llm_model = "llama3.2" #'deepseek-r1:1.5b' #"llama3.2" #llama3.1:latest
    llm = Ollama(model=llm_model, temperature=0.0, cache=False)
    

    system_prompt = """
        You are an AI specialized in sentiment analysis of **news transcripts**.  

        Analyze the provided financial news article and determine the **market sentiment** strictly based on **factual information**. Ignore linguistic tone, emotional cues, and speculative language—focus only on concrete financial indicators, company performance, and macroeconomic context.


        #### **Sentiment Analysis Criteria:**
            - **Sentiment**: Label as **"Positive"**, **"Neutral"**, or **"Negative"** based on financial data and market impact.
            - **Sentiment Score**: A numerical value between **-1** (strongly negative) and **+1** (strongly positive), with **0** being neutral.
            - **Key Factors**: List of **objective financial indicators** (e.g., earnings growth, revenue decline, stock reaction) driving sentiment.

        ### **Dictionary Output Format (Strictly Adhere to This Format)**:
            
            {{"Sentiment": "", # "Positive", "Neutral", or "Negative"
            "Sentiment_Score": "",    # Between -1 and +1 (floating-point number)
            "Key_Factors": [] # List of **keywords only** influencing sentiment
            }}
        Give the output in the above dictionary format without using any more words or explanations.
        
        Input:
        """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),  # Placeholder for user query
    ])
    
    # Create the LLM chain
    chain = prompt | llm
    
    # Invoke the chain with the input query
    response = chain.invoke({"input": query})
    
    return response
