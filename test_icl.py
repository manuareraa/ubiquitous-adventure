#!/usr/bin/env python3

import json
from agents.question_agent import QuestioningAgent

# Test ICL sample loading and matching
def test_icl_samples():
    print("=== Testing ICL Sample Loading ===")
    
    # Load the files
    inc_samples = QuestioningAgent.load_icl_samples("assets/topics_example.json")
    with open("assets/topics.json") as f: 
        topics = json.load(f)
    
    print(f"Loaded ICL samples for topics: {list(inc_samples.keys())}")
    print(f"Loaded topics structure: {topics}")
    
    # Test the key matching
    for main_topic, subtopics in topics.items():
        print(f"\nMain topic: {main_topic}")
        for subtopic in subtopics:
            print(f"  Subtopic: {subtopic}")
            if subtopic in inc_samples:
                print(f"    ✓ Found {len(inc_samples[subtopic])} ICL samples")
                # Show first example
                if inc_samples[subtopic]:
                    first_example = inc_samples[subtopic][0]
                    print(f"    First example question: {first_example['question'][:100]}...")
            else:
                print(f"    ✗ No ICL samples found for '{subtopic}'")
    
    # Test building ICL samples
    agent = QuestioningAgent()
    if "Truth-teller and Liar Problems" in inc_samples:
        samples = inc_samples["Truth-teller and Liar Problems"]
        icl_string = agent.build_inc_samples(samples, "Logical Reasoning/Truth-teller and Liar Problems")
        print(f"\n=== ICL String Preview ===")
        print(icl_string[:500] + "..." if len(icl_string) > 500 else icl_string)

if __name__ == "__main__":
    test_icl_samples()
