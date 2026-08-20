import re
from typing import List


STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "with", "from", "to", "of", "in", "on",
    "at", "by", "as", "is", "was", "were", "we", "they", "he", "she", "it", "this",
    "that", "are", "be", "will", "our", "their", "his", "her", "team", "meeting"
}

NON_PERSON_WORDS = {
    "Good", "Finally", "Once", "Today", "Tomorrow", "Monday", "Tuesday",
    "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Next",
    "The", "We", "It", "This", "That",
}


def clean_transcript(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> List[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def infer_deadline(sentence: str) -> str:
    lower = sentence.lower()
    patterns = [
        r"\b(?:by|on|before|after)\s+((?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?:\s+\d{1,2}(?:st|nd|rd|th)?)?(?:\s+[a-z]+)?(?:\s+\d{4})?)",
        r"\b(?:by|on|before|after)\s+((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?(?:\s+\d{4})?)",
        r"\b(?:by|on|before|after)\s+(\d{1,2}(?:st|nd|rd|th)?\s+[a-z]+(?:\s+\d{4})?)",
        r"\b(today|tomorrow)\b",
        r"\b(\d{1,2}:\d{2}(?:\s*[ap]m)?)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, lower, re.IGNORECASE)
        if match:
            return " ".join(part.capitalize() for part in match.group(1).strip(" .,!? ").split())
    return ""


def infer_owner(sentence: str) -> str:
    matches = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", sentence)
    owners = [name for name in matches if name.split()[0] not in NON_PERSON_WORDS]
    return owners[0] if owners else ""


def normalize_owner(owner: str, task: str) -> str:
    candidate = re.sub(r"\s+", " ", (owner or "").strip()).strip(" .,;:-")
    if candidate and re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}", candidate):
        if candidate.split()[0] not in NON_PERSON_WORDS:
            return candidate
    return infer_owner(task)


def extract_topics(text: str) -> List[str]:
    keywords = {
        "Project Updates": ["project", "progress", "timeline", "status", "roadmap"],
        "Design": ["design", "ui", "mockup", "prototype", "wireframe"],
        "Testing": ["test", "qa", "debug", "bug", "validation"],
        "Delivery": ["deadline", "launch", "submission", "handoff", "release"],
        "Collaboration": ["team", "roles", "handover", "review", "feedback"],
    }

    lower = clean_transcript(text).lower()
    topics = [label for label, terms in keywords.items() if any(term in lower for term in terms)]
    return topics[:4] if topics else ["Project Updates", "Action Items", "Risks", "Next Steps"]


def extract_decisions(text: str) -> List[str]:
    sentences = split_sentences(clean_transcript(text))
    decisions = []
    decision_markers = ["decided", "agreed", "approved", "confirmed", "will", "must", "need to", "should"]

    for sentence in sentences:
        lower = sentence.lower()
        if any(marker in lower for marker in decision_markers):
            if len(sentence) > 12 and not re.search(r"\b(will|need to|must|should)\b.*\b(prepare|review|test|fix|update)\b", lower):
                decisions.append(sentence)

    if not decisions:
        for sentence in sentences:
            if len(sentence) > 15:
                decisions.append(sentence)

    return decisions[:4]


def extract_action_items(text: str) -> List[List[str]]:
    sentences = split_sentences(clean_transcript(text))
    action_items = []
    action_markers = ["will", "must", "need to", "should", "can", "going to", "assigned to"]

    for sentence in sentences:
        lower = sentence.lower()
        if any(marker in lower for marker in action_markers):
            owner = infer_owner(sentence)
            deadline = infer_deadline(sentence)
            action_items.append([sentence, owner, deadline])

    if not action_items:
        for sentence in sentences[:3]:
            action_items.append([sentence, "", ""])

    return action_items[:5]

