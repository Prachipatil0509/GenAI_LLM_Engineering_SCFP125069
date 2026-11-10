# 🧪 Diagnostic Lab Test Booking Assistant (GenAI Assignment)

A production-ready GenAI diagnostic laboratory booking web application built with **Streamlit**, **LangChain**, **LangGraph**, **Groq LLM**, and **SQLite**.

---

## 🌟 Key Features

1. **Intelligent Conversational Agent**:
   - Queries real database records for test prices and slot availability.
   - Refuses to guess or hallucinate pricing or schedule slots.
   - Answers preparation and test description inquiries using live DuckDuckGo web search.

2. **Human-In-The-Loop (HITL) Safeguards**:
   - Integrated with LangChain's `HumanInTheLoopMiddleware`.
   - Modifying actions (`book_test`, `cancel_booking`) are paused automatically for user confirmation.
   - Interactive **Approve** / **Reject** UI cards directly inside the Streamlit chat flow.

3. **Privacy & PII Protection**:
   - Automatic regex email redaction ensures patient emails are masked (`[REDACTED EMAIL]`) before sending prompts to the LLM.

4. **Multi-Patient Context Isolation**:
   - Uses `ToolRuntime[UserContext]` to ensure patients can only see, book, and cancel their own appointments (`patient_101`, `patient_202`, etc.).

5. **Medical Ethics & Diagnosis Refusal**:
   - Strict system prompt guardrails: Refuses to diagnose symptoms or recommend tests for medical conditions; directs patients to qualified healthcare professionals.

6. **Live Database Inspector & Scenario Evaluator**:
   - Real-time viewer for `tests`, `slots`, and `bookings` tables in SQLite (`lab_booking.db`).
   - One-click Evaluation Scenario Suite matching the 8 test scenarios from the assignment evaluation matrix.

---

## 📁 Project Structure

```
GenAI_LLM_Engineering_SCFP125069/
├── app.py                            # Main Streamlit web frontend
├── agent_core.py                     # LangChain agent, tools, HITL middleware, and system prompt
├── db.py                             # SQLite database helpers, initialization, and live inspect queries
├── lab_booking.db                    # SQLite database file
├── Lab_Test_Booking_Assistant.ipynb  # Assignment notebook
├── requirements.txt                  # Project dependencies
├── .env.example                      # Environment variable template
└── README.md                         # Documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
Ensure Python 3.10+ is installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Streamlit Application
Navigate to this directory in your terminal:
```bash
cd "C:\Users\prach\OneDrive\Desktop\GenAI_LLM_Engineering_SCFP125069"
```

Run Streamlit:
```bash
streamlit run app.py
```

### 4. Provide your Groq API Key
- Enter your Groq API Key in the left sidebar (or set `GROQ_API_KEY` in your environment or a `.env` file).
- Select your preferred model (e.g. `openai/gpt-oss-20b` or `llama-3.3-70b-versatile`).

---

## 🧪 Evaluation Test Scenarios (Matrix from Assignment)

You can run each of these in the **Evaluation Test Suite** tab or type them in the chat:

| # | Scenario | Expected Tools | Expected DB Action | Test Prompt |
|---|---|---|---|---|
| **1** | Check availability | `list_tests`, `check_slots` | No change | *"List available diagnostic tests and check slots for Blood Sugar (Test ID 2)."* |
| **2** | Booking approved | `book_test` | New booking confirmed | *"Please book test ID 1 (CBC) in slot ID 1."* (Click Approve) |
| **3** | Duplicate booking | `book_test` | Prevented | *"Please book test ID 1 in slot ID 2."* (Already booked) |
| **4** | Booking rejected | `book_test` | No change | *"Book test ID 3 in slot ID 7."* (Click Reject) |
| **5** | Slot conflict / Unavailable | `check_slots` | No change | *"Check available appointment slots for test ID 8 (HbA1c)."* |
| **6** | Cancellation | `cancel_booking` | Booking cancelled, slot freed | *"Please cancel my confirmed booking."* |
| **7** | General information | `general_health_search` | No change | *"What does a Lipid Profile test measure?"* |
| **8** | Symptom question | None (Refusal) | No change | *"I have a severe headache and dizziness. Which test should I take?"* |

---

## 🛠️ Database Schema

- **`tests`**: `test_id` (PK), `test_name` (UNIQUE), `category`, `price`
- **`slots`**: `slot_id` (PK AUTOINCREMENT), `test_id` (FK), `slot_time`, `is_booked` (0 or 1)
- **`bookings`**: `booking_id` (PK AUTOINCREMENT), `user_id`, `test_id` (FK), `slot_id` (FK), `status` ('confirmed' / 'cancelled')
