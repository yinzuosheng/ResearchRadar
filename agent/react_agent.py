from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent.tools.agent_tools import (
    collect_and_ingest_topics,
    collect_and_ingest_papers,
    generate_brief,
    ingest_urls,
    push_brief,
    query_knowledge_base,
    search_papers,
    search_web,
)
from model.factory import chat_model
from utils.prompt_loader import load_system_prompt


class ReactAgent:
    def __init__(self) -> None:
        system_prompt = load_system_prompt()
        tools = [
            search_web,
            search_papers,
            collect_and_ingest_topics,
            collect_and_ingest_papers,
            ingest_urls,
            query_knowledge_base,
            generate_brief,
            push_brief,
        ]

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        agent = create_react_agent(chat_model, tools, prompt)
        self.executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    def run(self, query: str) -> str:
        result = self.executor.invoke({"input": query})
        return result.get("output", "")

    def execute_stream(self, query: str):
        for chunk in self.executor.stream({"input": query}):
            if "output" in chunk:
                yield chunk["output"]
