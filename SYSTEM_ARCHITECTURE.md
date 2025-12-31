# BankbotAI System Architecture

This document provides a comprehensive overview of the BankbotAI system, detailing its architecture, components, and the flow of control within the application.

## 1. System Overview
BankbotAI is an intelligent banking assistant built with Streamlit. It leverages the Lyzr AI API to provide users with a conversational interface for banking-related tasks, such as checking balances, comparing credit cards, viewing transaction history, and more. The system ensures persistence using a local SQLite database and implements security measures like rate limiting and input validation.

## 2. Core Components

### 2.1 UI Layer (Streamlit)
The frontend is built using **Streamlit**, which provides an interactive web interface.
- **`main()`**: The entry point of the application, responsible for setting up the page layout, sidebar, and handling user input/output.
- **Sidebar**: Provides navigation, chat history management (Search, Export, Delete), and AI service status.
- **Chat Interface**: A responsive chat window where users can interact with the bot.

### 2.2 Functional Services (`app.py`)

#### `DatabaseService`
Handles all interactions with the `chat_history.db` SQLite database.
- **Session Management**: Creates, archives, and deletes chat sessions.
- **Message Persistence**: Saves and loads messages for each session.
- **Search**: Allows searching through past chat messages.

#### `LyzrAPIService`
Manages communication with the Lyzr AI API.
- **Query Processing**: Sends user messages to the API and parses the response.
- **Retry Logic**: Implements an exponential backoff strategy for handling service interruptions.
- **Health Check**: Monitors the status of the AI service.

#### `BankingService`
Acts as a bridge between the intent detection and the actual data/AI response.
- **Structured Responses**: Generates formatted tables for specific intents like Credit Cards, ATMs, and Loans.
- **Caching**: Implements a simple MD5-based cache to reduce redundant API calls.

#### `IntentDetector`
A keyword-based classification engine that identifies the user's intent (e.g., `BALANCE`, `TRANSFER`, `LOAN`) to route the query to the appropriate handler.

#### `InputValidator` & `RateLimiter`
- **Validation**: Ensures user inputs are safe, not empty, and within length limits.
- **Rate Limiting**: Prevents abuse by limiting the frequency of requests per user.

## 3. Data Model

The system uses a SQLite database (`chat_history.db`) with two main tables:

| Table | Purpose |
| :--- | :--- |
| **`sessions`** | Stores chat session metadata (ID, title, timestamp, archived status). |
| **`messages`** | Stores individual messages linked to sessions (role, content, timestamp). |

## 4. Operational Flow Control

The following sequence describes how a user query is processed:

1.  **Input**: User enters a message in the chat input.
2.  **Validation**: `InputValidator` checks for length and suspicious patterns; `RateLimiter` checks if the user is allowed to send a message.
3.  **Persistence**: The user's message is saved to the database via `DatabaseService`.
4.  **Intent Detection**: `IntentDetector.detect()` analyzes the query to determine the category.
5.  **Response Generation**:
    - If the intent matches a **structured handler** (e.g., ATM list), `BankingService` returns a predefined table.
    - Otherwise, the query is sent to `LyzrAPIService`, which queries the **Lyzr AI API**.
6.  **Sanitization**: The bot's response is sanitized for safety.
7.  **Display**: The response is streamed or displayed directly in the Streamlit UI.
8.  **Final Persistence**: The assistant's response is saved to the database.

## 5. Technology Stack
- **Languages**: Python
- **Frontend**: Streamlit
- **Backend API**: Lyzr AI (studio.lyzr.ai)
- **Database**: SQLite3
- **Libraries**: `requests`, `pandas`, `logging`, `re`
