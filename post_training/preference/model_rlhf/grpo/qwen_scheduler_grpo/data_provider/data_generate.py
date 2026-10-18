# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_generate.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-05-04
# * Version     : 1.0.050421
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
import json
import random
from typing import Dict, List, Tuple


import datasets
from bisect import bisect_right

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]

"""
Event Scheduling Dataset Generator

This module generates synthetic event scheduling problems with the following characteristics:
- Random number of events (4-8) with varying durations
- Events can overlap with a controlled probability
- Some events are marked as priority events
- Each problem includes an optimal score calculation
"""

# Set seeds for reproducibility
random.seed(42)

# Event generation constants
MIN_EVENTS = 4
MAX_EVENTS = 8
DURATIONS = [15, 30, 45, 60, 75, 90, 105, 120]
MAX_START_HOUR = 21  # Ensures events finish within the day

# Overlap probability and constraints
OVERLAP_PROBABILITY = 0.2
MIN_OVERLAPS = 1  # Must have at least one overlap
MAX_OVERLAP_RATIO = 0.4  # Maximum 40% of events can overlap

# Priority selection constraints
MIN_PRIORITY_RATIO = 0.2  # Minimum 20% of events are priority
MAX_PRIORITY_RATIO = 0.4  # Maximum 40% of events are priority


def _get_events_categories_names(root_dir = None):
    """
    Load the events categories names
    """
    root_dir = "E:\\projects\llm_projects\\TinyLLM\\grpo\\qwen_scheduler_grpo\\dataset\\"
    data_dir = "events_categories_names.json"
    with open(Path(root_dir).joinpath(data_dir), "r") as file:
        events_categories_names = json.load(file)
    
    return events_categories_names


def _minutes_to_time(minutes: int) -> str:
    """
    Convert minutes since midnight to HH:MM time string.

    Args:
        minutes (int): Number of minutes since midnight

    Returns:
        str: Time in HH:MM format
    """
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _time_to_minutes(time_str: str) -> int:
    """
    Convert HH:MM time string to minutes since midnight.

    Args:
        time_str (str): Time in HH:MM format

    Returns:
        int: Number of minutes since midnight
    """
    hours, mins = map(int, time_str.split(":"))

    return hours * 60 + mins


def _count_overlapping_events(events):
    """
    Count the number of overlapping event pairs in a schedule.

    Args:
        events (list): List of events, where each event is a tuple of (name, start_time, end_time)

    Returns:
        int: Number of overlapping event pairs
    """
    overlapping_count = 0
    for j in range(len(events)):
        for k in range(j + 1, len(events)):
            # event 1
            e1_start = _time_to_minutes(events[j][1])
            e1_end = _time_to_minutes(events[j][2])
            # event 2
            e2_start = _time_to_minutes(events[k][1])
            e2_end = _time_to_minutes(events[k][2])
            # count overlap
            if e1_start < e2_end and e2_start <= e1_end:
                overlapping_count += 1
    
    return overlapping_count


def _random_event():
    """
    Generate a random event with random start time and duration.

    Returns:
        tuple: (start_time, end_time) in HH:MM format
    """
    # debug
    # start_mins = MAX_START_HOUR * 60 + 59
    # duration = max(DURATIONS)
    # end_mins = start_mins + duration
    # logger.info(f"debug::start_mins: {start_mins}, end_mins: {end_mins}")
    # debug
    start_mins = random.randint(0, MAX_START_HOUR * 60 + 59)
    duration = random.choice(DURATIONS)
    end_mins = start_mins + duration

    return _minutes_to_time(start_mins), _minutes_to_time(end_mins)


def _overlapping_event(prev_event):
    """
    Generate an event that overlaps with a previous event.

    Args:
        prev_event (tuple): Previous event tuple (name, start_time, end_time)

    Returns:
        tuple: (start_time, end_time) in HH:MM format
    """
    # prev event start and end
    prev_start_mins = _time_to_minutes(prev_event[1])
    prev_end_mins = _time_to_minutes(prev_event[2])
    # new event start and end
    start_mins = random.randint(prev_start_mins, prev_end_mins - 1)
    duration = random.choice(DURATIONS)
    end_mins = start_mins + duration

    return _minutes_to_time(start_mins), _minutes_to_time(end_mins)


def _generate_events() -> Tuple[List, List]:
    """
    Generate a valid schedule of events with controlled overlap and priority constraints.

    Returns:
        tuple: (events, priority_list) where:
            - events: List of (name, start_time, end_time) tuples
            - priority_list: List of event names marked as priority
    """
    # events categories
    events_categories_names = _get_events_categories_names()
    category = random.choice(list(events_categories_names.keys()))
    # Keey trying until we get a valid schedule
    while True:
        # create a copy
        event_names = list(events_categories_names[category])
        # data collect
        events = []
        priority = []

        # generate events
        n_events = random.randint(MIN_EVENTS, MAX_EVENTS)
        for i in range(1, n_events + 1):
            # choice event_name
            event_name = random.choice(event_names)
            event_names.remove(event_name)
            # generate event start and end
            if (i == 1) or (random.random() > OVERLAP_PROBABILITY) or (not events):
                event_start, event_end = _random_event()
            else:
                event_start, event_end = _overlapping_event(random.choice(events))
            # collect events info
            events.append((event_name, event_start, event_end))
            # events info sort
            events.sort(key = lambda x: x[1])
        
        # count total overlaps
        total_overlaps = _count_overlapping_events(events)

        # Check if we have a valid schedule
        if total_overlaps > MIN_OVERLAPS and total_overlaps < MAX_OVERLAP_RATIO * len(events):
            # select priority events
            min_priority = max(1, int(len(events) * MIN_PRIORITY_RATIO))
            max_priority = int(len(events) * MAX_PRIORITY_RATIO)
            n_priority = random.randint(min_priority, max_priority)
            priority_events = random.sample(events, n_priority)

            # priority_list
            priority_list = [e[0] for e in priority_events]
            # events
            events = sorted(events, key=lambda x: _time_to_minutes(x[1]))
    
            return events, priority_list


def _compute_optimal_score(events, priority_list):
    """
    Compute the optimal score for a schedule using dynamic programming.

    This implements a weighted interval scheduling algorithm where priority events
    have double the weight of regular events.
    We want to maximize the total weighted duration of the events.

    Inspired by: https://algo.monster/liteproblems/1235

    Args:
        events (list): List of (name, start_time, end_time) tuples
        priority_list (list): List of event names marked as priority

    Returns:
        int: Maximum possible score for the schedule
    """
    start_times = []
    end_times = []
    profits = []
    for event in events:
        start_times.append(_time_to_minutes(event[1]))
        end_times.append(_time_to_minutes(event[2]))
        weight = 2 if event[0] in priority_list else 1
        duration = _time_to_minutes(event[2]) - _time_to_minutes(event[1])
        profits.append(weight * duration)

    # Combine the job information into a single list and sort by end time.
    jobs = sorted(zip(end_times, start_times, profits))

    # Get the total number of jobs.
    number_of_jobs = len(jobs)

    # Initialize dynamic programming table with 0 profits.
    dp = [0] * (number_of_jobs + 1)

    # Iterate over the jobs.
    for i, (current_end_time, current_start_time, current_profit) in enumerate(jobs):
        # Find the rightmost job that doesn't conflict with the current job's start time.
        # Use binary search for efficient querying. 'hi' is set to the current index 'i' for optimization.
        index = bisect_right(jobs, current_start_time, hi=i, key=lambda x: x[0])

        # Update the DP table by choosing the maximum of either taking the current job or not.
        # If taking the current job, add its profit to the total profit of non-conflicting jobs.
        dp[i + 1] = max(dp[i], dp[index] + current_profit)

    # Return the maximum profit which is the last element in the DP table.
    return dp[number_of_jobs]


def _generate_now() -> Dict:
    """
    Generate a single scheduling problem with all required information.

    Returns:
        Dict: Dictionary containing:
            - events: List of events with times
            - priority_events: List of priority event names
            - optimal_score: Maximum possible score
            - prompt: Human-readable description of the problem
    """
    dict_events = {}

    events, priority_list = _generate_events()
    # events
    dict_events["events"] = events
    logger.info(f"debug::events: \n{events}")
    # priority list
    dict_events["priority_list"] = priority_list
    logger.info(f"debug::priority_list: \n{priority_list}")
    # optimal score
    dict_events["optimal_score"] = _compute_optimal_score(events, priority_list)
    logger.info(f"debug::optimal_score: \n{dict_events['optimal_score']}")
    # prompt
    prompt = (
        "Events:\n"
        + "\n".join([f"- {event[0]} ({event[1]} - {event[2]})" for event in events])
        + "\n\n"
    )
    prompt += "Priorities:\n" + "\n".join([f"- {priority}" for priority in priority_list])
    dict_events["prompt"] = prompt
    logger.info(f"debug::prompt: \n{prompt}")

    return dict_events


def generate_dataset():
    """
    Generate a complete dataset of scheduling problems and upload to Hugging Face.
    """
    dataset_list = []
    for _ in range(600):  # train: 500, test: 100
        dataset_list.append(_generate_now())
    # Dataset
    dataset = datasets.Dataset.from_list(dataset_list)
    # Dataset split
    dataset = dataset.train_test_split(test_size = 100, seed = 42)
    
    return dataset


def dataset_push_to_hub(dataset): 
    # uncomment to push the dataset
    # from huggingface_hub import notebook_login
    # notebook_login()
    dataset.push_to_hub("wangzf/events-scheduling")
    logger.info(f"Dataset has pushed to huggingface hub")




# 测试代码 main 函数
def main():
    dataset = generate_dataset()
    dataset_push_to_hub(dataset = dataset)

if __name__ == "__main__":
    main()
