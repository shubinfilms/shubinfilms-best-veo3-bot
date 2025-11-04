import asyncio
from types import SimpleNamespace

from handlers import video as video_handlers


class DummyBot:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append({"chat_id": chat_id, "text": text, "kwargs": kwargs})


def test_veo_semaphore_limits_parallel_jobs(monkeypatch):
    async def scenario():
        running = 0
        peak = 0

        async def fake_process(**kwargs):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1

        monkeypatch.setattr(video_handlers, "_process_animation_job", fake_process)

        bot = DummyBot()
        context = SimpleNamespace(bot=bot, chat_data={})

        tasks = [
            video_handlers._run_limited_animation_job(
                context=context,
                chat_id=1,
                user_id=123,
                job_id=f"job-{idx}",
                source_url="https://example.com/img.png",
                prompt="Prompt",
                progress=None,
            )
            for idx in range(video_handlers.MAX_CONCURRENT_VEO_TASKS + 2)
        ]

        await asyncio.gather(*tasks)

        assert peak <= video_handlers.MAX_CONCURRENT_VEO_TASKS
        queued_msgs = [m for m in bot.messages if "очереди" in m["text"]]
        assert queued_msgs, "Queue notice should be sent when semaphore is saturated"

    asyncio.run(scenario())
