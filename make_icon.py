from PIL import Image

# Open your PNG (make sure the file exists in the same folder)
img = Image.open("chatbot.png")

# Save as ICO with multiple sizes for Windows compatibility
img.save("chatbot.ico", format="ICO", sizes=[
    (16, 16), (32, 32), (48, 48),
    (64, 64), (128, 128), (256, 256)
])
