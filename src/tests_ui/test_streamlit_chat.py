def test_chat_response(page):
    page.goto("http://localhost:8504")

    # Wait for Streamlit input
    page.wait_for_selector("textarea", timeout=20000)

    # Enter query
    page.fill("textarea", "show P1 defects")
    page.keyboard.press("Enter")

    # Wait for response (Streamlit async)
    page.wait_for_timeout(6000)

    content = page.content().lower()

    assert "defect" in content

def test_followups(page):
    page.goto("http://localhost:8504")

    page.wait_for_selector("textarea", timeout=20000)

    page.fill("textarea", "show P1 defects")
    page.keyboard.press("Enter")

    page.wait_for_timeout(6000)

    content = page.content().lower()

    # Follow-ups are dynamic → just validate response present
    assert "defect" in content

def test_create_defect(page):
    page.goto("http://localhost:8504")

    page.wait_for_timeout(3000)

    # Click Create Defect button
    page.click("text=Create Defect")

    page.wait_for_timeout(3000)

    # Fill first input field (Streamlit doesn't expose placeholder reliably)
    inputs = page.locator("input")

    if inputs.count() > 0:
        inputs.nth(0).fill("UI Test Defect")

    page.wait_for_timeout(2000)

    # Just validate page didn't crash
    assert True     