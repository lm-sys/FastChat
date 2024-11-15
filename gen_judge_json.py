import json
import re

# Define the input and output file paths
input_file = "judge.txt"
output_file = "judge.json"

# Define a list to hold parsed entries
entries = []

# Open and read the input file
with open(input_file, "r") as file:
    content = file.read()

# Use regex to match each entry
entry_pattern = re.compile(
    r"Entry (?P<entry_number>\d+/\d+)\n"
    r"User Question: (?P<user_question>.+?)\n"
    r"A=(?P<a_label>.+?)'s Response: (?P<a_response>.+?)\n"
    r"B=(?P<b_label>.+?)'s Response: (?P<b_response>.+?)\n"
    r"Verdict: (?P<verdict>.+?)\n"
    r"Final Verdict: \[\[(?P<final_verdict>[AB])\]\]\n"
    r"Better response: (?P<better_response>.+)"
    , re.DOTALL)

# Find all entries that match the pattern
matches = entry_pattern.finditer(content)

# Process each match
for match in matches:
    entry = {
        "entry_number": match.group("entry_number"),
        "user_question": match.group("user_question").strip(),
        "assistant_A": {
            "label": match.group("a_label").strip(),
            "response": match.group("a_response").strip()
        },
        "assistant_B": {
            "label": match.group("b_label").strip(),
            "response": match.group("b_response").strip()
        },
        "verdict": match.group("verdict").strip(),
        "final_verdict": match.group("final_verdict"),
        "better_response": match.group("better_response").strip()
    }
    entries.append(entry)

# Write the entries to a JSON file with indentation and newlines preserved
with open(output_file, "w") as f:
    json.dump(entries, f, indent=4, ensure_ascii=False)

print("JSON file with better formatting created successfully.")
