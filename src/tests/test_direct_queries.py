from src.ai_chatbot.chatbot import ChatBot

bot = ChatBot()

queries = [
    "Show all P1 defects",
    "List defects assigned to SUBHRA",
    "Find defects with TCode ME21N",
    "Show defects from release R2",
    "Show closed defects",
    "List defects in progress",
    "show me existing open defects for TCode ME23N",
    "Show high priority defects assigned to ROSHA",
    "show me P1 defects from DLL opco that are Open"
]

def test_direct_queries():
    for q in queries:
        res = bot.ask_with_tokens(q)
        assert res.answer is not None
        assert len(res.answer) > 0