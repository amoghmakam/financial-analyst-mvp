import os
import subprocess
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from crewai import Agent, Task, Crew

from langchain.tools import Tool
from pydantic.v1 import BaseModel, Field


class RunCommandArgs(BaseModel):
    command: str = Field(..., description="Shell command to run from the project root")


def _run_command(command: str = None, tool_input=None, **kwargs) -> str:
    """
    CrewAI may call tools in different ways depending on version:
    - with a dict like {"command": "..."} (preferred)
    - with tool_input="..." (older style)
    - with command="..."
    This function supports all of them.
    """
    # If CrewAI passes a dict
    if isinstance(tool_input, dict) and not command:
        command = tool_input.get("command")

    # If CrewAI passes a string via tool_input
    if isinstance(tool_input, str) and not command:
        command = tool_input

    # If CrewAI passes dict in kwargs
    if not command and "command" in kwargs:
        command = kwargs["command"]

    if not command or not isinstance(command, str):
        raise ValueError("Missing command string. Provide {'command': '...'}")

    res = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),  # project root
        env=os.environ,
    )
    out = (res.stdout or "").strip()
    err = (res.stderr or "").strip()
    if res.returncode != 0:
        raise RuntimeError(
            f"Command failed ({res.returncode}): {command}\nSTDERR:\n{err}\nSTDOUT:\n{out}"
        )
    return out or "(ok)"


run_command_tool = Tool(
    name="run_command",
    func=_run_command,
    args_schema=RunCommandArgs,
    description="Run a shell command from the project root. Input must be an object: {'command': '...'}",
)


def main():
    load_dotenv(dotenv_path=".env")

    tickers = os.getenv("TICKERS", "AAPL,MSFT,NVDA,TSLA,AMZN")
    today = datetime.now().strftime("%Y-%m-%d")
    report_path = Path("reports") / f"daily_{today}.txt"

    scraper_agent = Agent(
        role="ScraperAgent",
        goal="Collect the latest SEC 8-K filings and produce cleaned docs.",
        backstory="You run the SEC scraper and cleaning pipeline reliably.",
        tools=[run_command_tool],
        verbose=True,
    )

    indexer_agent = Agent(
        role="IndexerAgent",
        goal="Update the FAISS index with any new chunks and embeddings.",
        backstory="You maintain the vector database so retrieval stays current.",
        tools=[run_command_tool],
        verbose=True,
    )

    analyst_agent = Agent(
        role="AnalystAgent",
        goal="Generate today's daily brief as a text report with source links.",
        backstory="You produce a concise analyst-style brief grounded in SEC filings.",
        tools=[run_command_tool],
        verbose=True,
    )

    t1 = Task(
        description=(
            f"Run the SEC scrape + clean pipeline for tickers: {tickers}.\n"
            "Use the run_command tool with object input.\n\n"
            f"1) {{'command': 'python scripts/scrape_sec.py --forms 8-K --limit 6 --tickers {tickers}'}}\n"
            "2) {'command': 'python preprocess/clean_docs.py'}\n"
            "Then confirm there are files in data/clean/sec."
        ),
        agent=scraper_agent,
        expected_output="Cleaned SEC docs exist in data/clean/sec.",
    )

    t2 = Task(
        description=(
            "Update the FAISS index incrementally.\n"
            "Use the run_command tool with object input.\n\n"
            "1) {'command': 'python embeddings/chunk_docs.py'}\n"
            "2) {'command': 'python embeddings/update_faiss.py'}\n"
            "Then confirm data/index/sec.index and data/index/sec_meta.json exist."
        ),
        agent=indexer_agent,
        expected_output="FAISS index updated successfully.",
    )

    t3 = Task(
        description=(
            "Generate today's daily brief.\n"
            "Use the run_command tool with object input.\n\n"
            "1) {'command': 'python automation/daily_report.py'}\n"
            f"Then confirm report exists at: {report_path}"
        ),
        agent=analyst_agent,
        expected_output=f"Daily report written to {report_path}.",
    )

    crew = Crew(
        agents=[scraper_agent, indexer_agent, analyst_agent],
        tasks=[t1, t2, t3],
        verbose=True,
    )

    result = crew.kickoff()
    print("\nCrew result:\n", result)
    print(f"\n✅ Report ready: {report_path}")


if __name__ == "__main__":
    main()
