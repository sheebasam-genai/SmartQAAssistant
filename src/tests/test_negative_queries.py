from src.ai_chatbot.chatbot import ChatBot

bot = ChatBot()

def test_negative_queries():
    queries = [
        "give me P3 defects in release R2",
        "give me P3 open defects in release R2"
    ]

    for q in queries:
        res = bot.ask_with_tokens(q)
        assert res.answer is not None   # should handle gracefully