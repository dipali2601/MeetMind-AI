SUMMARY_PROMPT = """
You are an assistant that turns meeting transcripts into concise project updates.
Summarize the transcript in 2 to 4 sentences.
Focus on key priorities, decisions, ownership, and delivery milestones.
Return only the summary text.
"""

DECISION_PROMPT = """
Extract the most important decisions from the meeting transcript.
Return only a JSON array of short decision statements.
"""

ACTION_PROMPT = """
Extract action items from the meeting transcript.
Return JSON with a list of objects:
[{"task": "...", "owner": "...", "deadline": "..."}]
Use an empty string if no person is named.
Owner must contain only a person's name and no extra words.
Deadline must contain only the deadline date, including weekday, day, month, and year when available.
Use an empty string when no deadline date is present.
"""
