from src.ai_chatbot.chatbot import ChatBot

bot = ChatBot()

def test_field_queries():
    queries = [
        "What's the root cause of COTEST-0002?",
        "Show me the resolution for COTEST-0003",
        "Which test cases are impacted by COTEST-0004?"
    ]

    for q in queries:
        res = bot.ask_with_tokens(q)
        assert res.answer != ""