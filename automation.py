from playwright.sync_api import sync_playwright, Page
from openai import OpenAI
from typing import Dict, Optional, List, Any
import logging
import json
import re
import base64
from bs4 import BeautifulSoup
import zxcvbn
import math
import time
import os 
import datetime
import uuid

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class ChatGPTAPI:
    def __init__(self, api_key: str, log_folder: str = None):
        self.client = OpenAI(api_key)
        self.model = "o3-mini-2025-01-31" # Izmantotais modelis
        self.log_folder = log_folder
        self.setup_logger()

    def setup_logger(self):
        os.makedirs("logs", exist_ok=True)
        
        self.logger = logging.getLogger("action-model")
        self.logger.setLevel(logging.INFO)
        
        log_file = os.path.join("logs", "deepseekFour.txt")
        file_handler = logging.FileHandler(log_file)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
        if self.log_folder:
            detailed_handler = logging.FileHandler(os.path.join(self.log_folder, "api_calls.log"))
            detailed_handler.setFormatter(formatter)
            self.logger.addHandler(detailed_handler)
        
    def log_interaction(self, input_text, output_text, elapsed_time):
        """Log the input, output and timing information"""
        self.logger.info(f"INPUT:\n{input_text}\n")
        self.logger.info(f"OUTPUT:\n{output_text}\n")
        self.logger.info(f"Time elapsed: {elapsed_time:.2f} seconds\n")
        
        timing_summary = f"Input tokens: {self.estimate_tokens(input_text)}, Output tokens: {self.estimate_tokens(output_text)}, Total tokens: {self.estimate_tokens(input_text) + self.estimate_tokens(output_text)}, Time: {elapsed_time:.2f}s"
        self.logger.info(timing_summary)
        
        if self.log_folder:
            with open(os.path.join(self.log_folder, "timing_summary.log"), 'a') as f:
                f.write(f"{timing_summary}\n")
                
        self.logger.info("-" * 80)

    def estimate_tokens(self, text: str) -> int:
        if text is None:
            return 0
        return len(str(text)) // 4
    
    def estimate_cost(self, messages: List[Dict[str, str]]) -> float:
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        estimated_tokens = total_chars / 4
        estimated_cost = (estimated_tokens / 1000) * 0.0015 + (estimated_tokens / 1000) * 0.002
        return estimated_cost
    
    def get_response(self, messages: List[Dict[str, str]], max_completion_tokens: int = 25000, json_only = False) -> str:
        try:
            estimated_cost = self.estimate_cost(messages)
            print(f"Estimated cost: ${estimated_cost:.4f}")
            
            start_time = time.time()
            
            if (json_only):
                print("Request JSON only response...")
                notJsonResponse = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=max_completion_tokens,
                )
                
                prompt_tokens = notJsonResponse.usage.prompt_tokens
                completion_tokens = notJsonResponse.usage.completion_tokens
                total_tokens = notJsonResponse.usage.total_tokens
                
                self.logger.info(f"Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
                
                notJsonResponse = notJsonResponse.choices[0].message.content
                
                print("Got Non JSON output handing over to GPT-4o-mini")
                jsonClient = OpenAI(api_key=api_key)
                jsonModel = "gpt-4o-mini"
                response = self.client.chat.completions.create(
                    model=jsonModel,
                    messages = [
                        {"role": "system", "content": "Parse the user command into JSON. "},
                        {"role": "user", "content": notJsonResponse}
                    ],
                    max_tokens=600,
                    temperature=0.7,
                    response_format={"type": "json_object"}
                )
                print("GPT-4o-mini completed parsing")
            else:
                print("Request initial response...")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=max_completion_tokens,
                )
                
            elapsed_time = time.time() - start_time
            
            input_text = json.dumps(messages, indent=2)
            
            self.log_interaction(input_text, response, elapsed_time)
                
            print("Parsed response object:")
            print(response)
                
            return response.choices[0].message.content
            
        except Exception as e:
            error_msg = str(e)
            if "Rate limit" in error_msg:
                return "Error: Rate limit exceeded. Please wait a moment before trying again."
            elif "Invalid API key" in error_msg or "Authentication" in error_msg:
                return "Error: Invalid API key. Please check your OpenAI API key."
            elif "Invalid request" in error_msg:
                return f"Error: Invalid request - {error_msg}"
            else:
                return f"Error: An unexpected error occurred - {error_msg}"

class WebAutomationAssistant:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.setup_logging()
        self.chatgpt = None
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler('automation.log'), logging.StreamHandler()]
        )
        self.logger = logging

    def create_log_folder(self, url: str, task: str):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        clean_url = re.sub(r'[^\w\-_]', '_', url.replace('https://', '').replace('http://', ''))[:50]
        
        clean_task = re.sub(r'[^\w\-_]', '_', task)[:50]
        
        folder_name = f"{timestamp}_{clean_url}_{clean_task}"
        
        log_folder = os.path.join("logs", "detailed", folder_name)
        os.makedirs(log_folder, exist_ok=True)
        
        os.makedirs(os.path.join(log_folder, "raw_html"), exist_ok=True)
        os.makedirs(os.path.join(log_folder, "cleaned_html"), exist_ok=True)
        
        self.logger.info(f"Created log folder: {log_folder}")
        return log_folder

    def is_noisy_string(self, string: str, threshold: float = 0.3) -> bool:
        """Implementation of Algorithm 1 from the Steward paper using zxcvbn."""
        if not string or not isinstance(string, str):
            return False
            
        if len(string) > 2 and len(string) < 73:
            result = zxcvbn.zxcvbn(string)
            num_guesses = result['guesses']          
            score = math.log2(num_guesses) / len(string)          
            contains_dictionary_words = any(match['dictionary_name'] is not None for match in result['sequence'] if 'dictionary_name' in match)          
            if not contains_dictionary_words and score > threshold:
                return True
                
        elif len(string) >= 73:
            return True
            
        return False

    def clean_html(self, html: str, task: str = None, log_folder: str = None, step: int = 0) -> str:
        if log_folder:
            with open(os.path.join(log_folder, "raw_html", f"step_{step}.html"), 'w', encoding='utf-8') as f:
                f.write(html)
                
        soup = BeautifulSoup(html, 'html.parser')
       
        original_element_count = len(soup.find_all())
       
        # 1. Solis: Interaktīvie elementi no Steward raksta
        steward_selectors = [
            "button", "a", "input", "select", "textarea", 
            "[role*=\"radio\"]", "[role*=\"option\"]", "[role*=\"checkbox\"]", 
            "[role*=\"button\"]", "[role*=\"tab\"]", "[role*=\"textbox\"]", 
            "[role*=\"link\"]", "[role*=\"menuitem\"]", "[role*=\"menu\"]", 
            "[role*=\"tabpanel\"]", "[role*=\"combobox\"]", "[role*=\"select\"]", 
            "[class*=\"radio\"]", "[class*=\"option\"]", "[class*=\"checkbox\"]", 
            "[class*=\"button\"]", "[class*=\"textbox\"]", "[class*=\"menuitem\"]", 
            "[class*=\"menu\"]", "[class*=\"tabpanel\"]", "[class*=\"combobox\"]", 
            "[class*=\"select\"]", "[class*=\"suggestion\"]", "[class*=\"search-bar\"]", 
            "[class*=\"search-result\"]", "[class*=\"toggle\"]", "[onclick]", 
            "[href]", "[aria-controls]", "[aria-label]", "[aria-labelledby]", 
            "[aria-haspopup]", "[aria-owns]", "[aria-selected]"
        ]
        
        interactable_elements = set()
        for selector in steward_selectors:
            try:
                matched_elements = soup.select(selector)
                interactable_elements.update(matched_elements)
            except Exception as e:
                self.logger.error(f"Error with selector '{selector}': {str(e)}")
        interactable_elements = list(interactable_elements)
        
        self.logger.info(f"Original elements: {original_element_count} | Interactable elements: {len(interactable_elements)}")
        
        # 2. Solis: Atslēgasvārdu pārbaude (testi norāda, ka soli varbūt labāk ir izlaist)
        filtered_elements = interactable_elements
        if task:
            task_lower = task.lower()
            keywords = set([word.lower() for word in re.findall(r'\b\w{3,}\b', task_lower)])           
            keywords.update(["search", "find", "go", "click", "enter", "type", "submit", "confirm", "book", "flight", "select"])        
            matched_elements = []
            for element in interactable_elements:
                element_text = element.get_text(strip=True).lower()
                element_attrs = " ".join([str(element.get(attr, "")) for attr in element.attrs]).lower()
                
                # Check if any keyword appears in the element text or attributes
                if any(keyword in element_text or keyword in element_attrs for keyword in keywords):
                    matched_elements.append(element)
                    
            filtered_elements = matched_elements if matched_elements else interactable_elements
            self.logger.info(f"Elements after keyword matching: {len(filtered_elements)}")
        
        # 3. Solis: Atribūtu tīrīšana
        important_attributes = [
            "id", "name", "class", "type", "value", "placeholder", 
            "href", "src", "alt", "title", "aria-label", "role",
            "for", "action", "method", "data-testid", "aria-expanded",
            "aria-controls", "aria-labelledby", "aria-haspopup", 
            "aria-owns", "aria-selected", "onclick", "data-date"
        ]
        
        cleaned_html = ""
        for i, element in enumerate(filtered_elements):
            cleaned_html += f"\n({i+1})\n"

            tag_name = element.name
            cleaned_tag = f"<{tag_name}"
            
            for attr_name, attr_value in element.attrs.items():
                if attr_name not in important_attributes:
                    continue
                    
                if isinstance(attr_value, list):
                    clean_values = [v for v in attr_value if not self.is_noisy_string(v)]
                    if clean_values:
                        attr_value = " ".join(clean_values)
                    else:
                        continue
                elif self.is_noisy_string(attr_value):
                    continue
                
                if isinstance(attr_value, str) and len(attr_value) > 100:
                    if attr_name in ["href", "src"]:
                        if "?" in attr_value:
                            attr_value = attr_value.split("?")[0] + "?..."
                    else:
                        attr_value = attr_value[:50] + "..."
                
                cleaned_tag += f' {attr_name}="{attr_value}"'
            
            cleaned_tag += ">"
            
            element_text = element.get_text(strip=True)
            if element_text:
                cleaned_tag += element_text
                
            cleaned_tag += f"</{tag_name}>"
            
            cleaned_html += cleaned_tag + "\n"
        
        original_size = len(html)
        cleaned_size = len(cleaned_html)
        reduction = ((original_size - cleaned_size) / original_size) * 100
        
        self.logger.info(f"HTML cleaning stats:")
        self.logger.info(f"Original size: {original_size:,} chars")
        self.logger.info(f"Cleaned size:  {cleaned_size:,} chars")
        self.logger.info(f"Reduction:     {reduction:.1f}%")
        self.logger.info(f"Est. tokens:   {int(cleaned_size / 4):,}")
        
        if log_folder:
            with open(os.path.join(log_folder, "cleaned_html", f"step_{step}.html"), 'w', encoding='utf-8') as f:
                f.write(cleaned_html)
        
        return cleaned_html

    def get_base64_screenshot(self, page: Page) -> str:
        """Take a screenshot and convert it to base64 for the LLM"""
        screenshot_bytes = page.screenshot()
        return base64.b64encode(screenshot_bytes).decode('utf-8')

    def get_action_sequence(self, page: Page, html: str, screenshot_base64: str, 
                           task: str, previous_actions: List[Dict], log_folder: str, step: int) -> List[Dict]:
        """Get a sequence of related actions from the LLM"""
        
        self.logger.info("FULL HTML:")
        self.logger.info(html)
        
        
        state = {
            "task": task,
            "current_url": page.url,
            "previous_actions": previous_actions,
            "cleaned_html": html,
        }
        
        log_state = state.copy()
        if 'screenshot_base64' in log_state:
            log_state['screenshot_base64'] = '[SCREENSHOT DATA]'
        
        print("Log state")
        print(log_state)
        
        # Vaicājums modelim:
        prompt = f"""
            You're a web automation assistant. Given:

            TASK: {task}
            URL: {page.url}
            ACTIONS: {json.dumps(previous_actions, indent=2)}
            HTML: {html}

            Return a JSON array with next action:
            [{{
              "action": "click|type|wait|done",
              "selector": "CSS selector for element",
              "value": "Text to type (for type actions only)"
            }}]

            Rules:
            - Use valid Playwright selectors
            - Mark task as done if goal appears complete
            - Don't repeat failed actions
            - If typing was last action, usually follow with click/press_enter
            - For completion: [{{"action": "done", "reason": "Task complete"}}]

            Return only valid JSON.
        """
        def estimate_tokens(text: str) -> int:
            return len(text) // 4
    
        MAX_TOKENS = 170000
        
        prompt_tokens = estimate_tokens(prompt)
        if prompt_tokens > MAX_TOKENS:
            excess_tokens = prompt_tokens - MAX_TOKENS
            excess_chars = excess_tokens * 4
            
            html_start = prompt.find("HTML: ") + 6
            
            current_html_length = len(html)
            new_html_length = current_html_length - excess_chars - 100
            
            if new_html_length > 1000:
                truncated_html = html[:new_html_length] + "\n<!-- HTML TRUNCATED TO FIT TOKEN LIMIT -->"
                
                prompt = f"""
                    You're a web automation assistant. Given:
                    TASK: {task}
                    URL: {page.url}
                    ACTIONS: {json.dumps(previous_actions, indent=2)}
                    HTML: {truncated_html}
                    Return a JSON array with next action:
                    [{{
                      "action": "click|type|wait|done",
                      "selector": "CSS selector for element",
                      "value": "Text to type (for type actions only)"
                    }}]
                    Rules:
                    - Use valid Playwright selectors
                    - Mark task as done if goal appears complete
                    - Don't repeat failed actions
                    - If typing was last action, usually follow with click/press_enter
                    - For completion: [{{"action": "done", "reason": "Task complete"}}]
                    Return only valid JSON.
                """
                
                self.logger.warning(f"HTML truncated from {current_html_length} to {new_html_length} characters to fit token limit")
    
        
        if log_folder:
            with open(os.path.join(log_folder, f"prompt_step_{step}.txt"), 'w', encoding='utf-8') as f:
                f.write(prompt)
        
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
            ],}
        ]
        
        print("Messages sent to GPT:")
        
        print(messages)
        
        response = self.chatgpt.get_response(messages, max_completion_tokens=25000, json_only=True)
        
        if log_folder:
            with open(os.path.join(log_folder, f"response_step_{step}.txt"), 'w', encoding='utf-8') as f:
                f.write(response)
        
        try:
            actions = json.loads(response)
            
            if "commands" in actions:
                actions = actions["commands"]
            elif isinstance(actions, list) and len(actions) > 0 and "commands" in actions[0]:
                actions = actions[0]["commands"]
            elif not isinstance(actions, list) or not all(isinstance(a, dict) and "action" in a for a in actions):
                return [{"action": "error", "reason": "Unexpected response format"}]
            
            print("actions obj")
            print(actions)
            
            return actions
            
        except json.JSONDecodeError:
            self.logger.error("Failed to parse action sequence as JSON")
            return [{"action": "error", "reason": "Failed to parse action sequence"}]

    def execute_action_sequence(self, page: Page, actions: List[Dict], debug_mode: bool = False) -> bool:
        """Execute a sequence of actions, stopping if the HTML significantly changes"""
        print("start execution sequence")
        if not actions or len(actions) == 0:
            self.logger.warning("No actions to execute")
            return False
                    
        for i, action in enumerate(actions):
            self.logger.info(f"Executing action {i+1}/{len(actions)}: {action.get('action')} {action.get('selector', '')}")
            
            if debug_mode:
                print(f"\nExecuting action {i+1}/{len(actions)}:")
                print(json.dumps(action, indent=2))
                choice = input("Execute this action? (y/n/q): ").lower()
                if choice == 'q':
                    print("Debugging session terminated by user")
                    return False
                elif choice == 'n':
                    print("Skipping action")
                    continue
            
            success = self.execute_action(page, action, debug_mode)
            
            if not success:
                self.logger.error(f"Failed to execute action: {action}")
                action["success"] = False
                action["failure_reason"] = "Couldnt perform selector selection! If its a cookie proposal or an email list, perhaps its no longer on screen and you can continue with the main task."
                return True
            
            action["success"] = True
                
            time.sleep(0.5)
            
        return True

    def execute_action(self, page: Page, action: Dict, debug_mode: bool = False) -> bool:
        """Execute a single automation action"""
        try:
            if action["action"] == "click":
                
                selectors = [
                    action["selector"],
                    f"text={action['selector']}",
                    f"[aria-label*=\"{action['selector']}\"]",
                    f"[title*='{action['selector']}']",
                    f"a[href*='{action['selector'].replace('a[href=', '').replace('\'', '').replace('"', '')}']",
                    "a:has-text('" + action['selector'].replace('a[href=', '').replace('/cinemas/', '').replace('\'', '').replace('"', '') + "')",
                    f"a:has-text('{action['selector']}')",
                    f"a >> text={action['selector']}",
                    ".navigation__link:has-text('" + action['selector'].replace('a[href=', '').replace('/cinemas/', '').replace('\'', '').replace('"', '') + "')",
                    ".navigation__link-text:has-text('" + action['selector'].replace('a[href=', '').replace('/cinemas/', '').replace('\'', '').replace('"', '') + "')"
                ]
                
                selectors = list(dict.fromkeys([s for s in selectors if s]))
                
                if debug_mode:
                    print("\nTrying selectors:")
                    for s in selectors:
                        print(f"- {s}")
                
                for selector in selectors:
                    try:
                        element = page.query_selector(selector)
                        if not element:
                            if debug_mode:
                                print(f"Selector not found: {selector}")
                            continue
                            
                        if not element.is_visible():
                            if debug_mode:
                                print(f"Element not visible: {selector}")
                            continue
                            
                        try:
                            element.scroll_into_view_if_needed()
                        except Exception as e:
                            if debug_mode:
                                print(f"Couldn't scroll to element: {selector} ({str(e)})")
                            pass
                            
                        page.wait_for_load_state('networkidle', timeout=5000)
                        
                        element.click(timeout=5000)
                        page.wait_for_load_state('networkidle', timeout=5000)
                        
                        self.logger.info(f"Successfully clicked element: {selector}")
                        return True
                        
                    except Exception as e:
                        if debug_mode:
                            print(f"Failed to click {selector}: {str(e)}")
                        continue
                
                self.logger.warning(f"Could not find clickable element. Tried selectors: {selectors}")
                return False

            elif action["action"] == "type":
                selectors = [
                    action["selector"],
                    f"input[placeholder*='{action['selector']}']",
                    f"input[aria-label*='{action['selector']}']",
                    f"textarea[placeholder*='{action['selector']}']"
                ]
                
                for selector in selectors:
                    try:
                        page.fill(selector, action["value"])
                        return True
                    except Exception:
                        continue
                        
                return False

            elif action["action"] == "wait" or action["action"] == "captcha":
                # page.wait_for_load_state('networkidle')
                return True

            elif action["action"] == "done":
                return True
                
            elif action["action"] == "press_enter":
                page.keyboard.press("Enter")
                    
                return True

            return False

        except Exception as e:
            self.logger.error(f"Action execution failed: {str(e)}")
            return False

    def run(self, task: str, start_url: str, max_steps: int = 10, debug_mode: bool = False) -> bool:
        """Main method to run the complete automation process with improvements"""
        log_folder = self.create_log_folder(start_url, task)
        
        self.chatgpt = ChatGPTAPI(self.api_key, log_folder)
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            previous_actions = []
            step_count = 0

            try:
                url = start_url
                page.goto(url)
                # page.wait_for_load_state('networkidle', timeout=5000)


                while step_count < max_steps:
                    if debug_mode:
                        print("\n" + "="*50)
                        print(f"Step {step_count + 1}/{max_steps}")
                        print(f"Current URL: {page.url}")
                        input("Press Enter to get next actions...")
                    time.sleep(5)

                    current_html = page.content()
                    print(bcolors.WARNING)
                    print(current_html)
                    print(bcolors.ENDC)
                    cleaned_html = self.clean_html(current_html, task, log_folder, step_count)
                    
                    print(bcolors.OKCYAN)
                    print(cleaned_html)
                    print(bcolors.ENDC)
                    
                    original_chars = len(current_html)
                    cleaned_chars = len(cleaned_html)
                    estimated_tokens = cleaned_chars / 4
                    
                    print(f"\nHTML Size Stats:")
                    print(f"Original: {original_chars:,} chars")
                    print(f"Cleaned:  {cleaned_chars:,} chars")
                    print(f"Estimated tokens: {int(estimated_tokens):,}")

                    screenshot_base64 = self.get_base64_screenshot(page)
                    
                    if log_folder:
                        screenshot_path = os.path.join(log_folder, f"screenshot_step_{step_count}.png")
                        with open(screenshot_path, "wb") as f:
                            f.write(base64.b64decode(screenshot_base64))
                    
                    action_sequence = self.get_action_sequence(
                        page, 
                        cleaned_html, 
                        screenshot_base64, 
                        task, 
                        previous_actions,
                        log_folder,
                        step_count
                    )
                    
                    if debug_mode:
                        print("\nProposed action sequence:")
                        print(json.dumps(action_sequence, indent=2))
                        choice = input("Execute this sequence? (y/n/q): ").lower()
                        if choice == 'q':
                            print("Debugging session terminated by user")
                            return False
                        elif choice == 'n':
                            print("Skipping sequence")
                            continue
                            
                    time.sleep(5)
                    
                    if len(action_sequence) == 1 and action_sequence[0]["action"] == "done":
                        self.logger.info("Task completed successfully")
                        return True
                        
                    if len(action_sequence) == 1 and action_sequence[0].get("action") == "error":
                        self.logger.error("Failed to determine actions")
                        return False

                    success = self.execute_action_sequence(page, action_sequence, debug_mode)
                    if not success:
                        self.logger.error("Failed to execute action sequence")
                        return False
                    
                    for action in action_sequence:
                        if "success" in action and (action["success"] == True or action["success"] == False): 
                            previous_actions.append(action)
                        
                    print(bcolors.OKCYAN)
                    print(previous_actions)
                    print(bcolors.ENDC)
                    
                    if log_folder:
                        with open(os.path.join(log_folder, f"actions_step_{step_count}.json"), 'w', encoding='utf-8') as f:
                            json.dump(action_sequence, f, indent=2)
                    
                    step_count += 1
                    
                if step_count >= max_steps:
                    self.logger.warning(f"Reached maximum step limit of {max_steps}")
                    return False

            except Exception as e:
                self.logger.error(f"Automation failed: {str(e)}")
                if log_folder:
                    with open(os.path.join(log_folder, "error.log"), 'w', encoding='utf-8') as f:
                        f.write(f"Automation failed: {str(e)}")
                return False

            finally:
                browser.close()

if __name__ == "__main__":
    
    test_cases = [
        {
            "url": "drugs.com",
            "task": "Show side effects of Tamiflu"
        },
    ]
   
    api_key = "your-api-key"
    
    assistant = WebAutomationAssistant(api_key)
    
    for test_case in test_cases:
        print(f"\n\nRunning test case: {test_case['url']} - {test_case['task']}")
        
        url = f"https://www.{test_case['url']}"
        task = test_case['task']
        
        try:
            success = assistant.run(
                task=task,
                start_url=url,
                max_steps=10, 
                debug_mode=False
            )
            
            print(f"Test completed: {'Success' if success else 'Failed'}")
        except Exception as e:
            print(f"Test error: {str(e)}")
            continue
    