from src.repository.defect_creator import DefectCreator

def test_create_defect():
    creator = DefectCreator(csv_path="test_defects.csv")

    data = {
        "Title": "Test defect",
        "Description": "Sample issue",
        "Severity": "P1",
        "OpCo": "DLL",
        "TCode": "ME21N",
        "Release": "R3.1.0"
    }

    success, msg, key = creator.create_defect(data)

    assert success is True
    assert key is not None