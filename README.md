# Course Scheduler Chatbot

This project is an AI-powered web application designed to help students build course schedules. It uses OpenAI's GPT models to understand user requests and generate personalized schedules based on available course data.

## Features

*   **Personalized Schedule Generation:** Creates course schedules based on student's year, major, secondary interests, and specific instructions.
*   **Course Data Integration:** Loads course details (schedule, area, level, instructor, grades) from Excel files for multiple departments.
*   **Schedule Updates:** Allows users to refine their generated schedule with follow-up requests.
*   **Conflict Detection & Resolution:** Identifies time conflicts in the schedule and attempts to resolve them automatically.
*   **Grade Distribution Analysis:** Displays grade distributions (A, B, C, etc.) and average GPA for courses in the schedule.
*   **Q&A:** Answers questions about courses, instructors, or the schedule using the loaded data.
*   **Save Schedule:** Allows users to download their final schedule as a JSON file.
*   **Web Interface:** Provides an interactive user interface built with Flask, Bootstrap, and jQuery.

## Technology Stack

*   **Backend:** Python, Flask
*   **AI Model:** OpenAI GPT-4o-mini (via openai library)
*   **Data Handling:** Pandas
*   **Frontend:** HTML, CSS, JavaScript, Bootstrap, jQuery
*   **Data Source:** Excel files (`.xlsx`)

## Setup and Installation

1.  **Prerequisites:**
    *   Python 3.x
    *   pip (Python package installer)

2.  **Clone the Repository (Optional):**
    *   If you have this project in a Git repository, clone it:
        ```bash
        git clone <repository-url>
        cd <repository-directory>
        ```

3.  **Install Dependencies:**
    *   Install the required Python libraries:
        ```bash
        pip install Flask openai pandas openpyxl
        ```
    *   *(Note: A `requirements.txt` file is recommended for larger projects)*

4.  **API Key Configuration:**
    *   You need an OpenAI API key.
    *   Open [`BotDefinition.py`](c:\Users\yunhu\Downloads\templates\BotDefinition.py) and replace `"Enter Your API Key Here"` with your actual key.
    *   Open [`GuiChatBot.py`](c:\Users\yunhu\Downloads\templates\GuiChatBot.py) and replace `"your key here"` with your actual key.
        ```python
        # In BotDefinition.py
        # self.client = OpenAI(api_key="YOUR_ACTUAL_OPENAI_API_KEY")

        # In GuiChatBot.py
        # chatbot = OpenAIBot("gpt-4o-mini", "YOUR_ACTUAL_OPENAI_API_KEY")
        ```

5.  **Data Files:**
    *   Ensure the course data Excel files (e.g., `courses for CS.xlsx`, `AAE.xlsx`, `BME.xlsx`, etc.) are present in the same directory as `GuiChatBot.py`. The application expects these files to load course information. See the `department_files` list in [`GuiChatBot.py`](c:\Users\yunhu\Downloads\templates\GuiChatBot.py) for the expected filenames.

## Running the Application

1.  Navigate to the project directory in your terminal.
2.  Run the Flask application:
    ```bash
    python GuiChatBot.py
    ```
3.  Open your web browser and go to `http://127.0.0.1:5000` (or the address provided in the terminal output).

## Usage

1.  **Enter Student Info:** Fill out the initial form with your name, year, major, optional secondary interest, and areas of interest within your major. Click "Continue".
2.  **Provide Instructions:** Describe the type of schedule you want (e.g., number of courses, specific topics, time preferences). Click "Generate Schedule".
3.  **Review Schedule:** The application will display the generated schedule, a summary, and any time conflicts.
4.  **Use Operations:**
    *   **Update Schedule:** Provide feedback or request changes to the current schedule.
    *   **Grade Distribution:** View grade statistics for the scheduled courses.
    *   **Ask Questions:** Ask specific questions about the courses or instructors.
    *   **Save Schedule:** Download the current schedule as a JSON file.
    *   **Resolve Conflicts:** If conflicts exist, attempt automatic resolution.
5.  **Navigate:** Use the "Back" buttons to return to previous steps.

## Testing

Refer to the [TESTING_GUIDELINES.md](c:\Users\yunhu\Downloads\README\README\TESTING_GUIDELINES.md) file for detailed steps on how to test the application using a specific user persona.

## File Structure

```
.
├── GuiChatBot.py             # Main Flask application logic
├── BotDefinition.py          # OpenAI Bot interaction class
├── templates/
│   └── course_scheduler.html # Frontend HTML template
├── *.xlsx                    # Course data files (e.g., courses for CS.xlsx)
├── README.md                 # This file
└── README/
    └── README/
        └── TESTING_GUIDELINES.md # Testing instructions
```
