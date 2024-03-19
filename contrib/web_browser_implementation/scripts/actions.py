from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

# Create a new instance of the Chrome driver and keep it open
def Create_a_new_instance_of_the_Chrome_driver(shared_variables, x: int):
    chrome_options = Options()
    chrome_options.add_experimental_option("detach", True)
    driver = webdriver.Chrome(options=chrome_options)
    shared_variables["driver"] = driver
    return {"Driver": shared_variables["driver"]}

# Navigate to a web page
def Navigate_to_a_web_page(shared_variables, url: str):
    driver = shared_variables["driver"]
    driver.get(url)

# "See" and retrieve only interactable elements of the web page
def See_the_web_page(driver: webdriver):
    elements = driver.find_elements(By.XPATH, "//*[not(@disabled)]")
    return elements

# Wait for the presence of a specific element
def Wait_for_the_presence_of_a_specific_element(driver: webdriver, element_id: str):
    element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, element_id))
    )
    return element

# Perform actions on the element
def Perform_actions_on_the_element(element):
    element.click()