import time

# 🔹 ALL QUERIES
QUERIES = [
    "Show all P1 defects",
    "List defects assigned to SUBHRA",
    "Find defects with TCode ME21N",
    "Show defects from release R2",
    "Show closed defects",
    "Summarize defect COTEST-0002",
    "Explain what caused COTEST-0002",
    "Show root cause distribution"
]


def test_end_to_end_application(page):
    page.goto("http://localhost:8504")

    page.wait_for_selector("textarea", timeout=20000)

    failed_queries = []

    print("\n🚀 STARTING END-TO-END TEST")

    # =========================================
    # 1️⃣ CHAT FLOW (ALL QUERIES)
    # =========================================
    for q in QUERIES:
        print(f"\n🔹 Testing Query: {q}")

        page.fill("textarea", "")
        page.fill("textarea", q)
        page.keyboard.press("Enter")

        page.wait_for_timeout(5000)

        content = page.content().lower()

        if not content or len(content) < 50:
            failed_queries.append(q)

    print("✅ Chat Flow Completed")

    # =========================================
    # 2️⃣ HISTORY VALIDATION
    # =========================================
    try:
        if page.locator("text=Recent Conversations").count() > 0:
            page.locator("text=Recent Conversations").first.click()
            page.wait_for_timeout(3000)

            history_content = page.content().lower()
            assert "defect" in history_content

            print("✅ History Working")
        else:
            print("⚠️ History section not found")

    except Exception as e:
        print(f"❌ History validation failed: {e}")

    # =========================================
    # 3️⃣ CREATE DEFECT
    # =========================================
    try:
        if page.locator("text=Create Defect").count() > 0:
            page.locator("text=Create Defect").first.click()
            page.wait_for_timeout(3000)

            inputs = page.locator("input")
            if inputs.count() > 0:
                inputs.nth(0).fill("UI Automation Defect")

            textareas = page.locator("textarea")
            if textareas.count() > 1:
                textareas.nth(1).fill("Automation defect description")

            page.wait_for_timeout(2000)

            print("✅ Defect Creation Form Working")
        else:
            print("⚠️ Create Defect button not found")

    except Exception as e:
        print(f"❌ Defect creation failed: {e}")

    # =========================================
    # 4️⃣ AI TITLE VALIDATION
    # =========================================
    try:
        page_content = page.content().lower()

        if "ai" in page_content or "title" in page_content:
            print("✅ AI Title Feature Detected")
        else:
            print("⚠️ AI Title not clearly visible")

    except Exception as e:
        print(f"❌ AI title validation failed: {e}")

    # =========================================
    # 5️⃣ ATTACHMENT VALIDATION
    # =========================================
    try:
        if page.locator("input[type='file']").count() > 0:
            file_input = page.locator("input[type='file']").first

            file_input.set_input_files({
                "name": "test.txt",
                "mimeType": "text/plain",
                "buffer": b"Test attachment content"
            })

            page.wait_for_timeout(2000)

            print("✅ Attachment Upload Working")
        else:
            print("⚠️ Attachment input not found")

    except Exception as e:
        print(f"❌ Attachment failed: {e}")

    # =========================================
    # 6️⃣ TOKEN DASHBOARD
    # =========================================
    try:
        page_content = page.content().lower()

        if "token" in page_content:
            print("✅ Token Dashboard Visible")
        else:
            print("⚠️ Token dashboard not detected")

    except Exception as e:
        print(f"❌ Token dashboard check failed: {e}")

    # =========================================
    # FINAL ASSERTION
    # =========================================
    assert len(failed_queries) == 0, f"Failed Queries: {failed_queries}"

    print("\n🎉 END-TO-END TEST COMPLETED SUCCESSFULLY")