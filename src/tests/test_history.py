from src.repository.history_manager import HistoryManager

def test_history_grouping():
    hm = HistoryManager("test_history.json")

    hm.save_conversation("q1", "a1", "session1")
    hm.save_conversation("q2", "a2", "session1")

    data = hm._read_data()

    assert len(data["conversations"]) == 1
    assert len(data["conversations"][0]["messages"]) == 2