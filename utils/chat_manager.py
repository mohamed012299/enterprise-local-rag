import os
import json
from datetime import datetime

# =========================================================
# CHAT HISTORY FOLDER
# =========================================================

CHAT_HISTORY_DIR = "chat_history"

if not os.path.exists(CHAT_HISTORY_DIR):

    os.makedirs(CHAT_HISTORY_DIR)

# =========================================================
# SAVE CHAT
# =========================================================

def save_chat(messages):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    filename = f"chat_{timestamp}.json"

    filepath = os.path.join(
        CHAT_HISTORY_DIR,
        filename
    )

    data = {
        "timestamp": timestamp,
        "messages": messages
    }

    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4
        )

    return filename

# =========================================================
# LOAD CHAT
# =========================================================

def load_chat(filename):

    filepath = os.path.join(
        CHAT_HISTORY_DIR,
        filename
    )

    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    return data

# =========================================================
# LIST CHATS
# =========================================================

def list_chats():

    files = os.listdir(
        CHAT_HISTORY_DIR
    )

    files = [
        f for f in files
        if f.endswith(".json")
    ]

    files.sort(reverse=True)

    return files