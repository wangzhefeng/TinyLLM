# -*- coding: utf-8 -*-

# ***************************************************
# * File        : reward_func.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-05-05
# * Version     : 1.0.050502
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import re
import random
from datetime import datetime


# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


"""
We use 3 reward functions:

Format reward: ensure the output is in the correct format. (10 points)
Sorted events reward: ensure the events are sorted in chronological order. (20 points)
Score reward: ratio between the total weighted duration of the events and the optimal score computed with dynamic programming. (70 points)
"""


def minutes_to_time(minutes):
    """
    Convert minutes since midnight to time string.

    Args:
        minutes (int): Minutes since midnight

    Returns:
        str: Time string in "HH:MM" format
    """
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def time_to_minutes(time_str):
    """
    Convert time string to minutes since midnight.

    Args:
        time_str (str): Time string in "HH:MM" format

    Returns:
        int: Minutes since midnight
    """
    hours, mins = map(int, time_str.split(":"))
    return hours * 60 + mins


# TODO
overall_pattern = r"<think>.+</think>.*<schedule>.*(<event>.*<name>.+</name>.*<start>\d{2}:\d{2}</start>.*<end>\d{2}:\d{2}</end>.*</event>)+.*</schedule>"
overall_regex = re.compile(overall_pattern, re.DOTALL)

# TODO
capture_pattern = r"""
    <event>\s*
        <name>([^<]+)</name>\s*
        <start>(\d{2}:\d{2})</start>\s*
        <end>(\d{2}:\d{2})</end>\s*
    </event>
"""
capture_regex = re.compile(capture_pattern, re.VERBOSE)


def get_events(content):
    """
    Extract event information from XML-like content.

    Args:
        content (str): XML-like string containing event data

    Returns:
        list: List of tuples (name, start_time, end_time)
    """
    return [
        (match.group(1), match.group(2), match.group(3))
        for match in capture_regex.finditer(content)
    ]


def format_reward(prompts, completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]

    return [
        0.0 if not overall_regex.match(response) else 10.0 for response in responses
    ]


def score_reward(prompts, completions, events, priority_events, optimal_score, **kwargs):
    scores = []
    responses = [completion[0]["content"] for completion in completions]
    for content, valid_events, priorities, opt_score in zip(responses, events, priority_events, optimal_score):
        scheduled_events = get_events(content)
        # Get valid scheduled events
        existing_events = {
            ev for ev in scheduled_events if [ev[0], ev[1], ev[2]] in valid_events
        }
        # penalize choosing nonexistent events or less than 2 events (not a valid schedule)
        if len(existing_events) < len(scheduled_events) or len(existing_events) < 2:
            scores.append(0.0)
            continue
        # Convert to minutes
        existing_events_minutes = [
            (ev[0], time_to_minutes(ev[1]), time_to_minutes(ev[2]))
            for ev in existing_events
        ]
        # remove overlapping events and remove both events - to penalize overlaps
        overlapping_events = set()
        for j in range(len(existing_events_minutes)):
            for k in range(j + 1, len(existing_events_minutes)):
                if (
                    existing_events_minutes[j][1] <= existing_events_minutes[k][2]
                    and existing_events_minutes[j][2] >= existing_events_minutes[k][1]
                ):
                    overlapping_events.add(existing_events_minutes[j])
                    overlapping_events.add(existing_events_minutes[k])
        existing_events_minutes = [
            ev for ev in existing_events_minutes if ev not in overlapping_events
        ]
        # Calculate score
        score = sum(
            2 * (ev[2] - ev[1]) if ev[0] in priorities else ev[2] - ev[1]
            for ev in existing_events_minutes
        )
        scores.append((score / opt_score) * 70)
    # Log samples
    if any(score > 0 for score in scores) and random.random() < 0.10:
        os.makedirs("completion_samples", exist_ok=True)
        log_file = Path("completion_samples").joinpath("completion_samples.txt")
        with open(log_file, "a") as f:
            f.write("\n\n==============\n")
            f.write(f"\n{datetime.now().time()}\n")
            f.write(f"{prompts[0]}\n")
            f.write(f"{scores}\n")
            f.write(f"{completions}")

    return scores


def sorted_events_reward(completions, **kwargs):
    scores = []
    responses = [completion[0]["content"] for completion in completions]
    for response in responses:
        scheduled_events = get_events(response)
        # not a valid schedule: should be discarded
        if len(scheduled_events) < 2:
            scores.append(0.0)
            continue
        scheduled_events_minutes = [
            (ev[0], time_to_minutes(ev[1]), time_to_minutes(ev[2]))
            for ev in scheduled_events
        ]
        if all(
            scheduled_events_minutes[i][1] < scheduled_events_minutes[i + 1][1]
            for i in range(len(scheduled_events_minutes) - 1)
        ):
            scores.append(20.0)
        else:
            scores.append(0)

    return scores




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
