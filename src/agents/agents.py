from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.tools import web_search, scrape_url
from dotenv import load_dotenv

load_dotenv()

llm = llm = ChatOllama(
    model="llama3-groq-tool-use",  # Specifically tuned for tool calling
    temperature=0,
)

# 1st agent :Search Agent
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search]
    )

# 2nd Agent :Reader Agent
def build_reader_agent():
    system_prompt = """You are a expert web scraping and information extraction agent.

YOUR TASK:
1. Pick the most relevant URL from the search results
2. Call scrape_url(url) to fetch the full content
3. Extract and return DETAILED, STRUCTURED information from that page - NOT just a brief summary

EXTRACTION RULES:
- Extract the FULL article/content, not just the first paragraph
- Include: page title, main heading(s), author (if present), date (if present)
- Extract all key facts, data points, statistics, numbers, and important information
- Keep bullet points, numbered lists, and tables intact
- For product pages: include price, features, specs, ratings, availability
- For news/articles: include full story, context, quotes, background
- For search/result pages: extract the most detailed content from the top result
- DO NOT truncate or overly summarize - preserve relevant details
- If the page has multiple sections, extract all of them

OUTPUT FORMAT (return as structured text):
Title: [page title]
URL: [the URL you scraped]
Summary: [2-4 sentence overview]
Key Points:
- [point 1]
- [point 2]
- [point 3]
Full Content:
[complete extracted content with headings, paragraphs, lists preserved]

IMPORTANT:
- DO NOT write any response before calling the tool - ONLY call scrape_url first
- After scraping, return the detailed extraction in the format above
- If no URL is provided in search results, ask for one"""

    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt=system_prompt
    )

#writer chain 

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be ain depth explainer and detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()




#critic_chain 

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()
