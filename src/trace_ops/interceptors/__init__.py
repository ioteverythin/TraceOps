"""Framework-specific interceptors for traceops.

These modules are loaded on-demand when a framework is detected.
If the framework isn't installed, the import is silently skipped.

Available interceptors:
  - langchain: Patches BaseChatModel.invoke / ainvoke, BaseTool.invoke / ainvoke
  - langgraph: Patches Pregel.invoke / ainvoke / stream / astream (graph-level events)
  - crewai: Patches Crew.kickoff, Agent.execute_task
"""
