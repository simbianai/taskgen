import os

def create_screenshots_folder():
    screenshots_folder = os.path.join(os.getcwd(), "screenshots")
    if not os.path.exists(screenshots_folder):
        os.makedirs(screenshots_folder)
    return screenshots_folder