import os
import re
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool, ToolRuntime
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from db import get_connection

@dataclass
class UserContext:
    user_id: str

# -------------------------------------------------------------
# PII Protection / Email Redaction
# -------------------------------------------------------------
def redact_email(message: str) -> str:
    """Mask email addresses from user prompt before sending to LLM."""
    pattern = r'[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}'
    return re.sub(pattern, "[REDACTED EMAIL]", message)

# -------------------------------------------------------------
# Database Tools
# -------------------------------------------------------------
@tool
def list_tests() -> str:
    """List all available diagnostic tests with category and price."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT test_id, test_name, category, price
        FROM tests
        ORDER BY test_id
    """).fetchall()
    conn.close()
    return str(rows)

@tool
def check_slots(test_id: int) -> str:
    """Check available appointment slots for a test ID."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT slot_id, slot_time
        FROM slots
        WHERE test_id = ? AND is_booked = 0
        ORDER BY slot_time
    """, (test_id,)).fetchall()
    conn.close()

    if not rows:
        return "No available slots for this test."
    return str(rows)

@tool
def my_bookings(runtime: ToolRuntime[UserContext]) -> str:
    """Show bookings belonging to the current patient."""
    user_id = runtime.context.user_id
    conn = get_connection()
    rows = conn.execute("""
        SELECT b.booking_id, t.test_name,
               s.slot_time, b.status
        FROM bookings b
        JOIN tests t ON b.test_id = t.test_id
        JOIN slots s ON b.slot_id = s.slot_id
        WHERE b.user_id = ?
        ORDER BY b.booking_id
    """, (user_id,)).fetchall()
    conn.close()

    return str(rows) if rows else "No bookings found."

@tool
def book_test(test_id: int, slot_id: int, runtime: ToolRuntime[UserContext]) -> str:
    """Book an available test appointment for the current patient."""
    user_id = runtime.context.user_id
    conn = get_connection()

    try:
        conn.execute("BEGIN IMMEDIATE")

        test = conn.execute(
            "SELECT test_name FROM tests WHERE test_id = ?",
            (test_id,)
        ).fetchone()

        if not test:
            conn.rollback()
            return "Invalid test ID."

        slot = conn.execute("""
            SELECT slot_time FROM slots
            WHERE slot_id = ? AND test_id = ? AND is_booked = 0
        """, (slot_id, test_id)).fetchone()

        if not slot:
            conn.rollback()
            return "This slot is unavailable."

        duplicate = conn.execute("""
            SELECT booking_id FROM bookings
            WHERE user_id = ? AND test_id = ?
            AND status = 'confirmed'
        """, (user_id, test_id)).fetchone()

        if duplicate:
            conn.rollback()
            return "You already have a confirmed booking for this test."

        cursor = conn.execute("""
            INSERT INTO bookings(user_id, test_id, slot_id)
            VALUES (?, ?, ?)
        """, (user_id, test_id, slot_id))

        conn.execute("""
            UPDATE slots SET is_booked = 1
            WHERE slot_id = ?
        """, (slot_id,))

        booking_id = cursor.lastrowid

        # Read receipt directly from database
        receipt = conn.execute("""
            SELECT b.booking_id, b.user_id, t.test_name,
                   t.price, s.slot_time, b.status
            FROM bookings b
            JOIN tests t ON b.test_id = t.test_id
            JOIN slots s ON b.slot_id = s.slot_id
            WHERE b.booking_id = ?
        """, (booking_id,)).fetchone()

        conn.commit()
        return f"Booking successful. Database receipt: {receipt}"

    except Exception as e:
        conn.rollback()
        return f"Booking failed: {str(e)}"

    finally:
        conn.close()

@tool
def cancel_booking(booking_id: int, runtime: ToolRuntime[UserContext]) -> str:
    """Cancel a booking belonging to the current patient."""
    user_id = runtime.context.user_id
    conn = get_connection()

    try:
        conn.execute("BEGIN IMMEDIATE")

        booking = conn.execute("""
            SELECT slot_id FROM bookings
            WHERE booking_id = ?
            AND user_id = ?
            AND status = 'confirmed'
        """, (booking_id, user_id)).fetchone()

        if not booking:
            conn.rollback()
            return "Booking not found or does not belong to you."

        slot_id = booking[0]

        conn.execute("""
            UPDATE bookings SET status = 'cancelled'
            WHERE booking_id = ? AND user_id = ?
        """, (booking_id, user_id))

        conn.execute("""
            UPDATE slots SET is_booked = 0
            WHERE slot_id = ?
        """, (slot_id,))

        conn.commit()
        return f"Booking {booking_id} cancelled successfully."

    except Exception as e:
        conn.rollback()
        return f"Cancellation failed: {str(e)}"

    finally:
        conn.close()

# -------------------------------------------------------------
# General Web Search Tool
# -------------------------------------------------------------
search = DuckDuckGoSearchRun()

@tool
def general_health_search(query: str) -> str:
    """Search general information about laboratory tests and preparation.
    Never use this tool for prices, availability or bookings."""
    return search.run(query)

# -------------------------------------------------------------
# System Prompt & Agent Setup
# -------------------------------------------------------------
SYSTEM_PROMPT = """
You are a diagnostic laboratory booking assistant.

RULES:

1. Use the database tools for all test prices.
2. Use the database tools for all appointment availability.
3. Never guess prices or available slots.
4. Always confirm the test and slot with the patient before booking.
5. Never book without human approval.
6. Never cancel without human approval.
7. Only show bookings belonging to the current user.
8. Use web search only for general health information.
9. Never diagnose diseases.
10. Never recommend or select a test based on symptoms.
11. For symptom-related questions, explain that you cannot diagnose
    and advise the patient to consult a qualified healthcare professional.
12. Provide booking receipts using database results.
"""

ALL_TOOLS = [
    list_tests,
    check_slots,
    my_bookings,
    book_test,
    cancel_booking,
    general_health_search
]

def build_agent(api_key: str, model_name: str = "openai/gpt-oss-20b", memory: InMemorySaver | None = None):
    """Factory to create the LangChain Lab Booking agent with HITL approval middleware."""
    if memory is None:
        memory = InMemorySaver()

    llm = ChatGroq(
        model=model_name,
        temperature=0,
        api_key=api_key
    )

    approval_middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "book_test": {
                "allowed_decisions": ["approve", "reject"]
            },
            "cancel_booking": {
                "allowed_decisions": ["approve", "reject"]
            }
        }
    )

    agent = create_agent(
        model=llm,
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        context_schema=UserContext,
        middleware=[approval_middleware],
        checkpointer=memory
    )

    return agent
