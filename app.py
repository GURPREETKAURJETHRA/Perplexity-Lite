from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_core.agents import AgentFinish
from langgraph.graph import END, Graph

from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.encoders import jsonable_encoder
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import base64

load_dotenv()
os.environ['OPENAI_API_KEY'] == os.getenv('OPENAI_API_KEY')
os.environ['TAVILY_API_KEY'] == os.getenv('TAVILY_API_KEY')


tools = [TavilySearchResults(max_results=1)]
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-4-1106-preview")

agent_runnable = create_openai_functions_agent(llm, tools, prompt)


agent = RunnablePassthrough.assign(
    agent_outcome=agent_runnable
)


def execute_tools(data):
    agent_action = data.pop('agent_outcome')
    tool_to_use = {t.name: t for t in tools}[agent_action.tool]
    observation = tool_to_use.invoke(agent_action.tool_input)
    data['intermediate_steps'].append((agent_action, observation))
    return data


def should_continue(data):
    if isinstance(data['agent_outcome'], AgentFinish):
        return "exit"
    else:
        return "continue"


workflow = Graph()

workflow.add_node("agent", agent)
workflow.add_node("tools", execute_tools)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",  # start node
    should_continue,
    {
        "continue": "tools",
        "exit": END
    }
)

workflow.add_edge('tools', 'agent')

chain = workflow.compile()


def make_serializable(obj):
    if isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(element) for element in obj]
    elif isinstance(obj, set):
        return list(obj)
    elif hasattr(obj, "__dict__"):
        # Convert custom objects to strings
        return str(obj)
    else:
        return obj


# chain.invoke({"input": "Tell me 5 startups in the field of Quantum Computing", "intermediate_steps": []})

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Endpoint integrating your workflow


@app.post("/process_query")
async def process_query(query: str = Form(...)):
    intermediate_steps = []
    raw_data = chain.invoke(
        {"input": query, "intermediate_steps": intermediate_steps})
    encoded_raw_data = base64.b64encode(str(raw_data).encode()).decode()
    print(raw_data)
    # Extract the desired output
    desired_output = raw_data["agent_outcome"].return_values['output']
    print(desired_output)
    # Extract the URL
    # Since the URL is in a list within a list, we navigate accordingly
    url = raw_data["intermediate_steps"][0][1][0]["url"]
    return JSONResponse(content={"raw_data": encoded_raw_data, "desired_output": desired_output, "url": url})
    # serializable_raw_data = make_serializable(raw_data)
    # return JSONResponse(content={"raw_data": json.dumps(raw_data)})
    # #return JSONResponse(content=jsonable_encoder(result))
