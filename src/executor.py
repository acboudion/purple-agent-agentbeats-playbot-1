from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
import logging

from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    Task,
    TaskState,
    UnsupportedOperationError,
    InvalidRequestError,
)
from a2a.utils.errors import ServerError
from a2a.utils import (
    new_agent_text_message,
    new_task,
)

from agent import Agent

logger = logging.getLogger(__name__)

# Shown to the caller when the agent raises unexpectedly. Deliberately generic: some evaluators
# (Pi-Bench) read a failed task's status text as an ordinary reply, so never leak exception text.
INTERNAL_ERROR_TEXT = "I ran into an internal problem handling that request. Please try again."


class BoundedTaskStore(InMemoryTaskStore):
    """InMemoryTaskStore that evicts the oldest tasks beyond `max_tasks`.

    Every Pi-Bench turn creates a new task whose history retains the full policy + transcript
    payload, and the SDK never evicts; a 71-scenario run would otherwise retain ~1000 copies.
    """

    def __init__(self, max_tasks: int = 256) -> None:
        super().__init__()
        self.max_tasks = max_tasks

    async def save(self, task: Task, context=None) -> None:
        async with self.lock:
            self.tasks.pop(task.id, None)  # re-insert so the dict keeps recency order
            self.tasks[task.id] = task
            while len(self.tasks) > self.max_tasks:
                del self.tasks[next(iter(self.tasks))]


TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected
}


class Executor(AgentExecutor):
    def __init__(self):
        self.agents: dict[str, Agent] = {} # context_id to agent instance

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg = context.message
        if not msg:
            raise ServerError(error=InvalidRequestError(message="Missing message in request"))

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(error=InvalidRequestError(message=f"Task {task.id} already processed (state: {task.status.state})"))

        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)

        context_id = task.context_id
        agent = self.agents.get(context_id)
        if not agent:
            agent = Agent()
            self.agents[context_id] = agent

        updater = TaskUpdater(event_queue, task.id, context_id)

        await updater.start_work()
        try:
            await agent.run(msg, updater)
            if not updater._terminal_state_reached:
                await updater.complete()
        except Exception:
            logger.exception("[%s] task %s failed with an unexpected agent error", context_id, task.id)
            await updater.failed(new_agent_text_message(INTERNAL_ERROR_TEXT, context_id=context_id, task_id=task.id))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())
