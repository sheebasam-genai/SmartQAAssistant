from src.ai_chatbot.chatbot import ChatBot

bot = ChatBot()

queries = [
    "Summarize defect COTEST-0002",
    "Explain what caused COTEST-0002",
    "Tell me about defect COTEST-0010",
    "Compare COTEST-0004, COTEST-0002, and COTEST-0003"
]

def test_llm_queries():
    for q in queries:
        res = bot.ask_with_tokens(q)
        assert res.answer != ""