# !/usr/bin/env python
# coding=utf-8

from enum import IntEnum

from rich.text import Text

__all__ = ["LogLevel", "AsyncMonitor"]

YELLOW_HEX = "#d4b702"


class LogLevel(IntEnum):
    OFF = -1  # No output
    ERROR = 0  # Only errors
    INFO = 1  # Normal output (default)
    DEBUG = 2  # Detailed output


class AsyncMonitor:
    def __init__(self, tracked_model, logger):
        self.step_durations = []
        self.tracked_model = tracked_model
        self.logger = logger
        if getattr(self.tracked_model, "last_input_token_count", "Not found") != "Not found":
            self.total_input_token_count = 0
            self.total_output_token_count = 0

    def get_total_token_counts(self):
        return {
            "input": self.total_input_token_count,
            "output": self.total_output_token_count,
        }

    def reset(self):
        self.step_durations = []
        self.total_input_token_count = 0
        self.total_output_token_count = 0

    async def update_metrics(self, step_log):
        """비동기로 모니터의 메트릭을 업데이트합니다.

        Args:
            step_log: Step log로, duration, input/output token 정보 포함.
        """
        step_duration = step_log.duration
        self.step_durations.append(step_duration)
        console_outputs = f"[Step {len(self.step_durations)}: Duration {step_duration:.2f} seconds"
        if getattr(self.tracked_model, "last_input_token_count", None) is not None:
            self.total_input_token_count += self.tracked_model.last_input_token_count
            self.total_output_token_count += self.tracked_model.last_output_token_count
            console_outputs += (
                f" | Input tokens: {self.total_input_token_count:,} | Output tokens: {self.total_output_token_count:,}"
            )
        console_outputs += "]"
        self.logger.log(Text(console_outputs, style="dim"), level=LogLevel.INFO)
