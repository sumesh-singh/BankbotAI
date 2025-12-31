import streamlit as st
import pandas as pd
import requests
import time
import sqlite3
import json
import re
import hashlib
import logging
from datetime import datetime, timedelta
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import os
from enum import Enum


class Config:
    """Application configuration with environment variable support"""
    LYZR_API_URL = 'https://agent-prod.studio.lyzr.ai/v3/inference/chat/'
    LYZR_API_KEY = 'YOUR API KEY'
    LYZR_USER_ID = 'YOUR MAIL'
    LYZR_AGENT_ID = 'your agent id'
    LYZR_SESSION_ID = 'ypur agent session id'
    DB_PATH = os.getenv('DB_PATH', 'chat_history.db')
    MAX_MESSAGE_LENGTH = 2000
    RATE_LIMIT_SECONDS = 1
    MAX_RETRIES = 3
    CACHE_TTL = 3600
    MESSAGES_PER_PAGE = 50
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')


logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bankbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IntentType(Enum):
    """User intent categories"""
    BALANCE = "balance"
    CREDIT_CARD = "credit_card"
    TRANSFER = "transfer"
    TRANSACTION = "transaction"
    ATM = "atm"
    LOAN = "loan"
    SAVINGS = "savings"
    GENERAL = "general"

@dataclass
class Message:
    """Message data structure"""
    role: str
    content: Any
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class ResponseData:
    """Structured response data"""
    type: str  # 'text', 'table', 'chart'
    text: str
    data: Optional[Dict] = None
    metadata: Optional[Dict] = None


class DatabaseService:
    """Handle all database operations with proper connection management"""
    
    def __init__(self, db_path: str = Config.DB_PATH):
        """
        Initialize database service.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()
        logger.info(f"Database initialized at {db_path}")
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections with automatic commit/rollback.
        
        Yields:
            sqlite3.Connection: Database connection
        """
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database schema with indexes"""
        with self.get_connection() as conn:
            c = conn.cursor()
            
            c.execute('''CREATE TABLE IF NOT EXISTS sessions
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                          title TEXT NOT NULL, 
                          timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                          archived BOOLEAN DEFAULT 0)''')
            
            c.execute("PRAGMA table_info(sessions)")
            columns = [info[1] for info in c.fetchall()]
            if 'archived' not in columns:
                logger.info("Migrating database: Adding archived column to sessions table")
                try:
                    c.execute("ALTER TABLE sessions ADD COLUMN archived BOOLEAN DEFAULT 0")
                except Exception as e:
                    logger.error(f"Migration failed: {e}")
            
            c.execute('''CREATE TABLE IF NOT EXISTS messages
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                          session_id INTEGER NOT NULL, 
                          role TEXT NOT NULL, 
                          content TEXT NOT NULL, 
                          timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                          FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE)''')
            
            c.execute('''CREATE INDEX IF NOT EXISTS idx_messages_session 
                         ON messages(session_id)''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_messages_timestamp 
                         ON messages(timestamp)''')
            c.execute('''CREATE INDEX IF NOT EXISTS idx_sessions_timestamp 
                         ON sessions(timestamp)''')
    
    def create_session(self, first_message: str) -> int:
        """
        Create a new chat session.
        
        Args:
            first_message: Initial message to generate title from
            
        Returns:
            ID of newly created session
        """
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                title = self._generate_title(first_message)
                c.execute("INSERT INTO sessions (title) VALUES (?)", (title,))
                session_id = c.lastrowid
                logger.info(f"Created session {session_id}: {title}")
                return session_id
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    def save_message(self, session_id: int, role: str, content: Any) -> bool:
        """
        Save a message to the database.
        
        Args:
            session_id: Session ID
            role: Message role (user/assistant)
            content: Message content (can be dict or string)
            
        Returns:
            True if successful
        """
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                content_str = json.dumps(content) if isinstance(content, dict) else content
                c.execute(
                    "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)", 
                    (session_id, role, content_str)
                )
                return True
        except Exception as e:
            logger.error(f"Failed to save message: {e}")
            return False
    
    def load_messages(self, session_id: int, limit: int = None) -> List[Message]:
        """
        Load messages for a session.
        
        Args:
            session_id: Session ID
            limit: Maximum number of messages to load
            
        Returns:
            List of Message objects
        """
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                query = "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp"
                if limit:
                    query += f" LIMIT {limit}"
                c.execute(query, (session_id,))
                rows = c.fetchall()
                
                messages = []
                for row in rows:
                    try:
                        content = json.loads(row['content'])
                    except (json.JSONDecodeError, TypeError):
                        content = row['content']
                    messages.append(Message(
                        role=row['role'],
                        content=content,
                        timestamp=datetime.fromisoformat(row['timestamp'])
                    ))
                return messages
        except Exception as e:
            logger.error(f"Failed to load messages: {e}")
            return []
    
    def get_all_sessions(self, archived: bool = False) -> List[Tuple[int, str, str]]:
        """
        Get all sessions.
        
        Args:
            archived: Whether to include archived sessions
            
        Returns:
            List of (id, title, timestamp) tuples
        """
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                query = "SELECT id, title, timestamp FROM sessions WHERE archived = ? ORDER BY timestamp DESC"
                c.execute(query, (1 if archived else 0,))
                return c.fetchall()
        except Exception as e:
            logger.error(f"Failed to get sessions: {e}")
            return []
    
    def delete_session(self, session_id: int) -> bool:
        """
        Delete a session and its messages.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if successful
        """
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                c.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                c.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                logger.info(f"Deleted session {session_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return False
    
    def search_messages(self, query: str) -> List[Tuple[int, str, str]]:
        """
        Search messages across all sessions.
        
        Args:
            query: Search query
            
        Returns:
            List of (session_id, title, matching_content) tuples
        """
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                c.execute("""
                    SELECT DISTINCT s.id, s.title, m.content 
                    FROM sessions s 
                    JOIN messages m ON s.id = m.session_id 
                    WHERE m.content LIKE ? 
                    ORDER BY m.timestamp DESC 
                    LIMIT 10
                """, (f'%{query}%',))
                return c.fetchall()
        except Exception as e:
            logger.error(f"Failed to search messages: {e}")
            return []
    
    def archive_session(self, session_id: int) -> bool:
        """Archive a session"""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                c.execute("UPDATE sessions SET archived = 1 WHERE id = ?", (session_id,))
                return True
        except Exception as e:
            logger.error(f"Failed to archive session: {e}")
            return False
    
    @staticmethod
    def _generate_title(message: str, max_length: int = 30) -> str:
        """Generate a title from a message"""
        title = message.strip()
        if len(title) > max_length:
            title = title[:max_length] + "..."
        return title


class IntentDetector:
    """Detect user intent from queries"""
    
    INTENT_KEYWORDS = {
        IntentType.BALANCE: ['balance', 'account', 'money', 'funds', 'available'],
        IntentType.CREDIT_CARD: ['credit card', 'compare', 'apr', 'card', 'credit'],
        IntentType.TRANSFER: ['transfer', 'send', 'pay', 'payment', 'wire'],
        IntentType.TRANSACTION: ['transaction', 'history', 'spent', 'purchase', 'statement'],
        IntentType.ATM: ['atm', 'cash', 'withdraw', 'location', 'nearest'],
        IntentType.LOAN: ['loan', 'mortgage', 'borrow', 'interest rate', 'eligibility'],
        IntentType.SAVINGS: ['savings', 'save', 'interest', 'deposit', 'investment'],
    }
    
    @classmethod
    def detect(cls, query: str) -> IntentType:
        """
        Detect intent from user query.
        
        Args:
            query: User query string
            
        Returns:
            Detected IntentType
        """
        query_lower = query.lower()
        
        for intent, keywords in cls.INTENT_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                logger.info(f"Detected intent: {intent.value}")
                return intent
        
        return IntentType.GENERAL


class LyzrAPIService:
    """Handle Lyzr API interactions with retry logic"""
    
    def __init__(self):
        """Initialize Lyzr API service"""
        self.api_url = Config.LYZR_API_URL
        self.api_key = Config.LYZR_API_KEY
        self.user_id = Config.LYZR_USER_ID
        self.agent_id = Config.LYZR_AGENT_ID
        self.session_id = Config.LYZR_SESSION_ID
        self.session = requests.Session()
        logger.info("Lyzr API Service initialized")
    
    def query(self, message: str, max_retries: int = Config.MAX_RETRIES) -> str:
        """
        Query Lyzr API with retry logic.
        
        Args:
            message: User message to send
            max_retries: Maximum number of retry attempts
            
        Returns:
            Response text from API
        """
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': self.api_key
        }
        
        payload = {
            'user_id': self.user_id,
            'agent_id': self.agent_id,
            'session_id': self.session_id,
            'message': message
        }
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Sending request to Lyzr API (attempt {attempt + 1})")
                response = self.session.post(
                    self.api_url, 
                    headers=headers, 
                    json=payload, 
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = self._parse_response(response.json())
                    logger.info(f"Lyzr API query successful on attempt {attempt + 1}")
                    return result
                else:
                    logger.warning(f"Lyzr API returned status {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Lyzr API request failed (attempt {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    return "I'm having trouble connecting to my AI service. Please try again later."
        
        return "Service temporarily unavailable. Please try again."
    
    def _parse_response(self, response_json: Dict) -> str:
        """
        Parse Lyzr API response and extract the message.
        
        Args:
            response_json: JSON response from API
            
        Returns:
            Extracted message text
        """
        try:
            
            if 'message' in response_json:
                return str(response_json['message'])
            
            if 'response' in response_json:
                return str(response_json['response'])
            
            if 'data' in response_json and isinstance(response_json['data'], dict):
                if 'message' in response_json['data']:
                    return str(response_json['data']['message'])
                if 'response' in response_json['data']:
                    return str(response_json['data']['response'])
            
            if 'choices' in response_json and len(response_json['choices']) > 0:
                choice = response_json['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    return str(choice['message']['content'])
                if 'text' in choice:
                    return str(choice['text'])
            
            if 'result' in response_json:
                return str(response_json['result'])
            
            logger.warning(f"Unknown response structure: {response_json}")
            return json.dumps(response_json, indent=2)
            
        except Exception as e:
            logger.error(f"Error parsing Lyzr API response: {e}")
            return f"Received response but couldn't parse it properly. Raw: {str(response_json)[:200]}"
    
    def check_health(self) -> bool:
        """Check if Lyzr API service is available"""
        try:
            response = self.session.post(
                self.api_url,
                headers={
                    'Content-Type': 'application/json',
                    'x-api-key': self.api_key
                },
                json={
                    'user_id': self.user_id,
                    'agent_id': self.agent_id,
                    'session_id': self.session_id,
                    'message': 'hello'
                },
                timeout=10
            )
            return response.status_code == 200
        except:
            return False


class BankingService:
    """Handle banking operations and data"""
    
    def __init__(self, lyzr_service: LyzrAPIService):
        """
        Initialize banking service.
        
        Args:
            lyzr_service: LyzrAPIService instance
        """
        self.lyzr = lyzr_service
        self.cache = {}
    
    def get_response(self, query: str, intent: IntentType) -> ResponseData:
        """
        Get banking response based on intent.
        
        Args:
            query: User query
            intent: Detected intent
            
        Returns:
            ResponseData object
        """
        if intent in [IntentType.CREDIT_CARD, IntentType.TRANSACTION, IntentType.ATM, IntentType.LOAN]:
            handlers = {
                IntentType.CREDIT_CARD: self._handle_credit_card,
                IntentType.TRANSACTION: self._handle_transaction,
                IntentType.ATM: self._handle_atm,
                IntentType.LOAN: self._handle_loan,
            }
            return handlers[intent](query)
        
        return self._handle_with_lyzr(query, intent)
    
    def _handle_with_lyzr(self, query: str, intent: IntentType) -> ResponseData:
        """Handle query using Lyzr API"""
        cache_key = hashlib.md5(query.encode()).hexdigest()
        
        if cache_key in self.cache:
            logger.info("Returning cached response")
            return self.cache[cache_key]
        
        response_text = self.lyzr.query(query)
        
        result = ResponseData(
            type="text",
            text=response_text,
            metadata={"intent": intent.value, "cached": False}
        )
        
        self.cache[cache_key] = result
        return result
    
    def _handle_credit_card(self, query: str) -> ResponseData:
        """Handle credit card comparisons"""
        data = {
            "Card Name": ["Silver Rewards", "Gold Travel Elite", "Platinum Business"],
            "APR": ["14.5%", "18.2%", "12.0%"],
            "Annual Fee": ["$0", "$95", "$250"],
            "Rewards": ["1% Cash Back", "2x Miles", "2% Cash Back"],
            "Sign-up Bonus": ["$150", "50K Miles", "$500"]
        }
        return ResponseData(
            type="table",
            text="💳 **Credit Card Comparison**\n\nHere are our top credit card options:",
            data=data,
            metadata={"intent": "credit_card"}
        )
    
    def _handle_transaction(self, query: str) -> ResponseData:
        """Handle transaction history"""
        data = {
            "Date": ["Dec 15", "Dec 14", "Dec 13", "Dec 12", "Dec 11"],
            "Description": ["Grocery Store", "Gas Station", "Online Payment", "Restaurant", "ATM Withdrawal"],
            "Amount": ["-$85.42", "-$45.00", "-$129.99", "-$67.50", "-$100.00"],
            "Balance": ["$4,250.00", "$4,335.42", "$4,380.42", "$4,510.41", "$4,577.91"]
        }
        return ResponseData(
            type="table",
            text="📊 **Recent Transactions**\n\nHere are your latest 5 transactions:",
            data=data,
            metadata={"intent": "transaction"}
        )
    
    def _handle_atm(self, query: str) -> ResponseData:
        """Handle ATM location requests"""
        data = {
            "Location": ["Main St Branch", "Downtown Plaza", "Shopping Mall", "Airport Terminal"],
            "Distance": ["0.3 miles", "0.8 miles", "1.2 miles", "3.5 miles"],
            "Available": ["24/7", "24/7", "Mall Hours", "24/7"],
            "Services": ["Deposit + Withdraw", "Withdraw Only", "Deposit + Withdraw", "Withdraw Only"]
        }
        return ResponseData(
            type="table",
            text="🏧 **Nearest ATM Locations**\n\nATMs near you:",
            data=data,
            metadata={"intent": "atm"}
        )
    
    def _handle_loan(self, query: str) -> ResponseData:
        """Handle loan inquiries"""
        data = {
            "Loan Type": ["Personal Loan", "Home Mortgage", "Auto Loan", "Business Loan"],
            "Interest Rate": ["7.5% - 12%", "6.2% - 7.8%", "5.5% - 9%", "8% - 15%"],
            "Max Amount": ["$50,000", "$500,000", "$75,000", "$250,000"],
            "Term": ["1-5 years", "15-30 years", "3-7 years", "1-10 years"]
        }
        return ResponseData(
            type="table",
            text="🏠 **Loan Options**\n\nExplore our loan products:",
            data=data,
            metadata={"intent": "loan"}
        )


class InputValidator:
    """Validate and sanitize user inputs"""
    
    @staticmethod
    def validate_message(message: str) -> Tuple[bool, str]:
        """
        Validate user message.
        
        Args:
            message: User input message
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not message or not message.strip():
            return False, "Message cannot be empty"
        
        if len(message) > Config.MAX_MESSAGE_LENGTH:
            return False, f"Message too long (max {Config.MAX_MESSAGE_LENGTH} characters)"
        
        suspicious_patterns = ['<script', 'javascript:', 'onerror=']
        if any(pattern in message.lower() for pattern in suspicious_patterns):
            return False, "Invalid characters detected"
        
        return True, ""
    
    @staticmethod
    def sanitize_output(text: str) -> str:
        """Sanitize output text"""
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        return text


class RateLimiter:
    """Simple rate limiter for user requests"""
    
    def __init__(self, limit_seconds: int = Config.RATE_LIMIT_SECONDS):
        """
        Initialize rate limiter.
        
        Args:
            limit_seconds: Minimum seconds between requests
        """
        self.limit_seconds = limit_seconds
        self.last_request = {}
    
    def check_limit(self, user_id: str) -> Tuple[bool, float]:
        """
        Check if user is within rate limit.
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (is_allowed, time_remaining)
        """
        now = datetime.now()
        
        if user_id in self.last_request:
            time_diff = (now - self.last_request[user_id]).total_seconds()
            if time_diff < self.limit_seconds:
                return False, self.limit_seconds - time_diff
        
        self.last_request[user_id] = now
        return True, 0


def stream_text(text: str):
    """Stream text token by token"""
    tokens = re.split(r'(\s+)', text)
    for token in tokens:
        yield token
        time.sleep(0.02)

def export_chat_json(messages: List[Message], session_id: int) -> str:
    """Export chat history as JSON"""
    data = {
        "session_id": session_id,
        "export_date": datetime.now().isoformat(),
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in messages
        ]
    }
    return json.dumps(data, indent=2)

def format_timestamp(dt: datetime) -> str:
    """Format timestamp for display"""
    now = datetime.now()
    diff = now - dt
    
    if diff.days == 0:
        if diff.seconds < 60:
            return "Just now"
        elif diff.seconds < 3600:
            return f"{diff.seconds // 60}m ago"
        else:
            return f"{diff.seconds // 3600}h ago"
    elif diff.days == 1:
        return "Yesterday"
    else:
        return dt.strftime("%b %d, %Y")


def init_services():
    """Initialize all services"""
    if 'services_initialized' not in st.session_state:
        st.session_state.db_service = DatabaseService()
        st.session_state.lyzr_service = LyzrAPIService()
        st.session_state.banking_service = BankingService(st.session_state.lyzr_service)
        st.session_state.rate_limiter = RateLimiter()
        st.session_state.services_initialized = True
        logger.info("Services initialized")

def init_session_state():
    """Initialize session state variables"""
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_search" not in st.session_state:
        st.session_state.show_search = False

def main():
    """Main application entry point"""
    
    st.set_page_config(
        page_title="BankBot AI",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
        /* Modern Reset & Typography */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            color: #e2e8f0;
            letter-spacing: -0.011em;
        }
        
        /* Premium Gradient Background */
        .stApp {
            background: linear-gradient(135deg, #0a0f1e 0%, #1a1f35 50%, #0f1729 100%);
            background-attachment: fixed;
        }

        /* Advanced Animations */
        @keyframes slideUpFade {
            from { 
                opacity: 0; 
                transform: translateY(30px) scale(0.95); 
            }
            to { 
                opacity: 1; 
                transform: translateY(0) scale(1); 
            }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes gradientShift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        
        @keyframes shimmer {
            0% { background-position: -200% center; }
            100% { background-position: 200% center; }
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }

        /* Enhanced Header with Animated Gradient */
        .main-header { 
            font-size: 4rem; 
            font-weight: 800; 
            background: linear-gradient(-45deg, #60a5fa 0%, #a78bfa 25%, #f472b6 50%, #fb923c 75%, #60a5fa 100%);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center; 
            margin-top: 3rem;
            margin-bottom: 0.75rem;
            letter-spacing: -0.04em;
            animation: slideUpFade 1s cubic-bezier(0.16, 1, 0.3, 1), gradientShift 8s ease infinite;
            filter: drop-shadow(0 0 30px rgba(96, 165, 250, 0.3));
        }
        
        .sub-header { 
            font-size: 1.15rem; 
            color: #94a3b8; 
            text-align: center; 
            margin-bottom: 4rem; 
            font-weight: 400;
            animation: slideUpFade 1s cubic-bezier(0.16, 1, 0.3, 1) 0.15s backwards;
            opacity: 0.85;
        }
        
        /* Premium Chat Messages with Enhanced Glass */
        .stChatMessage {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(30, 41, 59, 0.3) 100%);
            backdrop-filter: blur(16px) saturate(180%);
            -webkit-backdrop-filter: blur(16px) saturate(180%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 1.25rem;
            padding: 1.75rem;
            margin-bottom: 1.25rem;
            animation: slideUpFade 0.6s cubic-bezier(0.16, 1, 0.3, 1);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }
        
        .stChatMessage:hover {
            border-color: rgba(255, 255, 255, 0.12);
            box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.3);
            transform: translateY(-2px);
        }
        
        [data-testid="stChatMessageContent"] {
            color: #e2e8f0;
            line-height: 1.8;
            font-size: 0.95rem;
        }
        
        /* Enhanced Sidebar with Premium Glass */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(15, 23, 42, 0.90) 100%);
            backdrop-filter: blur(20px) saturate(180%);
            -webkit-backdrop-filter: blur(20px) saturate(180%);
            border-right: 1px solid rgba(96, 165, 250, 0.1);
            box-shadow: 4px 0 24px rgba(0, 0, 0, 0.2);
        }
        
        [data-testid="stSidebar"] .stMarkdown h3 {
            color: #f1f5f9;
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 1rem;
        }
        
        /* Premium Button Design with Gradients */
        .stButton button {
            position: relative;
            background: linear-gradient(135deg, rgba(96, 165, 250, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
            backdrop-filter: blur(10px);
            border: 1.5px solid rgba(96, 165, 250, 0.2);
            color: #f1f5f9;
            border-radius: 1.25rem;
            padding: 1rem 1.25rem;
            font-weight: 600;
            font-size: 0.9rem;
            width: 100%;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            text-align: center;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            animation: fadeIn 0.5s ease-out backwards;
            overflow: hidden;
        }
        
        .stButton button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
            transition: left 0.5s;
        }
        
        .stButton button:hover::before {
            left: 100%;
        }
        
        .stButton button:hover {
            background: linear-gradient(135deg, rgba(96, 165, 250, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%);
            border-color: rgba(96, 165, 250, 0.5);
            color: #ffffff;
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 12px 24px -4px rgba(96, 165, 250, 0.3), 
                        0 0 24px rgba(96, 165, 250, 0.2),
                        inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }
        
        .stButton button:active {
            transform: translateY(-1px) scale(0.99);
        }

        /* Enhanced Chat Input */
        [data-testid="stChatInput"] {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(30, 41, 59, 0.4) 100%);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(96, 165, 250, 0.2);
            border-radius: 1.5rem;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
        }
        
        [data-testid="stChatInput"]:focus-within {
            border-color: rgba(96, 165, 250, 0.5);
            box-shadow: 0 4px 32px rgba(96, 165, 250, 0.25);
        }
        
        /* Enhanced Text Inputs */
        .stTextInput input {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(30, 41, 59, 0.5) 100%) !important;
            backdrop-filter: blur(8px) !important;
            color: #f1f5f9 !important;
            border: 1.5px solid rgba(96, 165, 250, 0.15) !important;
            border-radius: 1rem !important;
            padding: 0.75rem 1rem !important;
            font-size: 0.9rem !important;
            transition: all 0.3s ease !important;
        }
        
        .stTextInput input:focus {
            border-color: rgba(96, 165, 250, 0.5) !important;
            box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.15), 
                        0 4px 12px rgba(96, 165, 250, 0.2) !important;
            background: rgba(30, 41, 59, 0.8) !important;
        }
        
        /* Toggle Styling */
        .stCheckbox {
            color: #cbd5e1;
        }

        /* Dataframe Styling */
        .stDataFrame {
            background: rgba(30, 41, 59, 0.4);
            backdrop-filter: blur(8px);
            border-radius: 0.75rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
            overflow: hidden;
        }

        /* Hide Streamlit Branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {
            visibility: visible !important;
            background: transparent !important;
        }
        [data-testid="stDecoration"] {
            display: none;
        }
        
        /* Refined Timestamps */
        .timestamp {
            font-size: 0.7rem;
            color: #64748b;
            margin-top: 0.5rem;
            font-weight: 500;
            letter-spacing: 0.02em;
        }
        
        /* Scrollbar Styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(15, 23, 42, 0.5);
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(96, 165, 250, 0.3);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(96, 165, 250, 0.5);
        }
    </style>
    """, unsafe_allow_html=True)
    
    init_services()
    init_session_state()
    
    db = st.session_state.db_service
    banking = st.session_state.banking_service
    rate_limiter = st.session_state.rate_limiter
    
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2534/2534204.png", width=40)
        st.markdown("### MyBank")
        
        if st.session_state.lyzr_service.check_health():
            st.caption("🟢 System Online")
        else:
            st.caption("🟡 System Check")
        
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            st.session_state.current_session_id = None
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # Search Toggle
        if st.toggle("Show Search", value=st.session_state.show_search):
             st.session_state.show_search = True
        else:
             st.session_state.show_search = False

        if st.session_state.show_search:
            search_query = st.text_input("Find messages...", key="search_input", label_visibility="collapsed")
            if search_query:
                results = db.search_messages(search_query)
                if results:
                    st.caption(f"Found {len(results)} results")
                    for i, (sess_id, title, content) in enumerate(results):
                        # Truncate content for button label
                        display_text = f"{title}"
                        if st.button(display_text, key=f"search_{sess_id}_{i}", help=content[:100]):
                            st.session_state.current_session_id = sess_id
                            st.session_state.messages = db.load_messages(sess_id)
                            st.session_state.show_search = False
                            st.rerun()
        
        st.markdown("### History")
        
        sessions = db.get_all_sessions()
        
        if sessions:
            for sess_id, title, timestamp in sessions:
                col1, col2 = st.columns([0.85, 0.15])
                with col1:
                    if st.button(f"{title}", key=f"sess_{sess_id}", use_container_width=True):
                        st.session_state.current_session_id = sess_id
                        st.session_state.messages = db.load_messages(sess_id)
                        st.rerun()
                with col2:
                    if st.button("x", key=f"del_{sess_id}", help="Delete"):
                        if db.delete_session(sess_id):
                            if st.session_state.current_session_id == sess_id:
                                st.session_state.current_session_id = None
                                st.session_state.messages = []
                            st.rerun()
        else:
            st.caption("No history yet")
    
    if not st.session_state.messages:
        # Minimal Home Page
        st.markdown('<div class="main-header">BankBot AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">How can I help you today?</div>', unsafe_allow_html=True)
        
        # Spacer
        st.markdown(" <br> ", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        initial_prompt = None
        
        with col1:
            if st.button("💰 Check Balance", use_container_width=True):
                initial_prompt = "What is Team B's current balance?"
            if st.button("📊 Transactions", use_container_width=True):
                initial_prompt = "Result transactions"
        
        with col2:
            if st.button("💸 Transfer Money", use_container_width=True):
                initial_prompt = "I want to transfer money"
            if st.button("🏠 Loan Options", use_container_width=True):
                initial_prompt = "Show me loan options"

        if initial_prompt:
            new_id = db.create_session(initial_prompt)
            st.session_state.current_session_id = new_id
            
            st.session_state.messages.append({
                "role": "user", 
                "content": initial_prompt,
                "timestamp": datetime.now()
            })
            db.save_message(new_id, "user", initial_prompt)
            
            intent = IntentDetector.detect(initial_prompt)
            bot_response = banking.get_response(initial_prompt, intent)
            
            response_data = {
                "type": bot_response.type,
                "text": bot_response.text,
                "data": bot_response.data
            }
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_data,
                "timestamp": datetime.now()
            })
            db.save_message(new_id, "assistant", response_data)
            st.rerun()
    
    else:
        for idx, message in enumerate(st.session_state.messages):
            if hasattr(message, 'role'):  # It's a Message object
                role = message.role
                content = message.content
                timestamp = message.timestamp
            else:  # It's a dict
                role = message.get("role")
                content = message.get("content")
                timestamp = message.get("timestamp")
                
            with st.chat_message(role):

                
                if isinstance(content, dict):
                    if content.get("type") == "table" and content.get("data"):
                        st.markdown(content["text"])
                        df = pd.DataFrame(content["data"])
                        st.dataframe(df, hide_index=True, use_container_width=True)
                    else:
                        st.markdown(content.get("text", content))
                else:
                    st.markdown(content)
                
                if timestamp:
                    ts = timestamp
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts)
                    st.markdown(
                        f'<div class="timestamp">{format_timestamp(ts)}</div>', 
                        unsafe_allow_html=True
                    )
    
    if prompt := st.chat_input("Ask me anything about your finances..."):
        is_valid, error_msg = InputValidator.validate_message(prompt)
        if not is_valid:
            st.error(f"❌ {error_msg}")
            return
        
        user_id = "default_user"  # In production, use actual user ID
        is_allowed, time_remaining = rate_limiter.check_limit(user_id)
        
        if not is_allowed:
            st.warning(f"⏳ Please wait {time_remaining:.1f} seconds before sending another message.")
            return
        
        if st.session_state.current_session_id is None:
            st.session_state.current_session_id = db.create_session(prompt)
        
        session_id = st.session_state.current_session_id
        
        user_msg = {
            "role": "user", 
            "content": prompt,
            "timestamp": datetime.now()
        }
        st.session_state.messages.append(user_msg)
        db.save_message(session_id, "user", prompt)
        
        with st.chat_message("user"):
            st.markdown(prompt)
            st.markdown(
                f'<div class="timestamp">{format_timestamp(user_msg["timestamp"])}</div>', 
                unsafe_allow_html=True
            )
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                intent = IntentDetector.detect(prompt)
                
                bot_response = banking.get_response(prompt, intent)
                
                bot_response.text = InputValidator.sanitize_output(bot_response.text)
                
                response_data = {
                    "type": bot_response.type,
                    "text": bot_response.text,
                    "data": bot_response.data
                }
                
                if bot_response.type == "table" and bot_response.data:
                    st.markdown(bot_response.text)
                    df = pd.DataFrame(bot_response.data)
                    st.dataframe(df, hide_index=True, use_container_width=True)
                else:
                    if hasattr(st, "write_stream"):
                        st.write_stream(stream_text(bot_response.text))
                    else:
                        st.markdown(bot_response.text)
                
                response_time = datetime.now()
                st.markdown(
                    f'<div class="timestamp">{format_timestamp(response_time)}</div>', 
                    unsafe_allow_html=True
                )
                
                assistant_msg = {
                    "role": "assistant", 
                    "content": response_data,
                    "timestamp": response_time
                }
                st.session_state.messages.append(assistant_msg)
                db.save_message(session_id, "assistant", response_data)
        
        logger.info(f"Session {session_id}: User query processed - Intent: {intent.value}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Application error: {e}", exc_info=True)
        st.error("🚨 An unexpected error occurred. Please refresh the page.")
        st.exception(e)