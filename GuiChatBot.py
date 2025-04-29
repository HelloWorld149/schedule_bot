# Web interface for the course scheduling chatbot

# Importing Libraries
from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import json
import time
import os
from BotDefinition import OpenAIBot

# Create Flask app with session support
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Set a secret key for session

# Initialize OpenAI bot with API key
chatbot = OpenAIBot("gpt-4o-mini", "your key here")

# Department abbreviation mapping
dept_abbr = {
    "Computer Science": "CS",
    "Aeronautics and Astronautics Engineering": "AAE",
    "Agricultural and Biological Engineering": "ABE",
    "Biomedical Engineering": "BME",
    "Civil Engineering": "CE",
    "Chemical Engineering": "CHE",
    "Electrical and Computer Engineering": "ECE",
    "Environmental and Ecological Engineering": "EEE",
    "Industrial Engineering": "IE",
    "Mechanical Engineering": "ME",
    "Material Science and Engineering": "MSE",
    "Nuclear Engineering": "NE"
}

# Helper function
def format_course_number(course_num, subject):
    """Format course number with department prefix if not already present."""
    course_num_str = str(course_num)

    # Handle potential float representation like "205.0"
    if course_num_str.endswith(".0"):
        course_num_str = course_num_str[:-2]

    # Check if the course number already has an alphabetic prefix
    if any(c.isalpha() for c in course_num_str):
        return course_num_str

    # Get the abbreviation for the subject
    abbr = dept_abbr.get(subject, "")
    if not abbr:
        # Try to derive abbreviation from the subject name if not in mapping
        words = subject.split()
        if words:
            abbr = ''.join(word[0] for word in words if word)

    # Combine abbreviation with course number
    if abbr:
        return f"{abbr}{course_num_str}"

    return course_num_str

# Try to load the courses dataset from multiple departments
try:
    department_files = [
        "courses for CS.xlsx",
        "AAE.xlsx",
        "ABE.xlsx",
        "BME.xlsx",
        "CE.xlsx",
        "CHE.xlsx",
        "ECE.xlsx",
        "EEE.xlsx",
        "IE.xlsx",
        "ME.xlsx",
        "MSE.xlsx",
        "nuclear_eng.xlsx"
    ]

    # Combine all department data
    all_dataframes = []
    
    # Department to subject mapping
    dept_to_subject = {
        "courses for CS.xlsx": "Computer Science",
        "AAE.xlsx": "Aeronautics and Astronautics Engineering",
        "ABE.xlsx": "Agricultural and Biological Engineering",
        "BME.xlsx": "Biomedical Engineering",
        "CE.xlsx": "Civil Engineering",
        "CHE.xlsx": "Chemical Engineering",
        "ECE.xlsx": "Electrical and Computer Engineering",
        "EEE.xlsx": "Environmental and Ecological Engineering",
        "IE.xlsx": "Industrial Engineering",
        "ME.xlsx": "Mechanical Engineering",
        "MSE.xlsx": "Material Science and Engineering",
        "nuclear_eng.xlsx": "Nuclear Engineering"
    }
    
    for file in department_files:
        try:
            df = pd.read_excel(file)
            
            # If the file exists, add the Subject column based on the filename
            if file in dept_to_subject:
                df["Subject"] = dept_to_subject[file]
            else:
                df["Subject"] = "Unknown"
                
            all_dataframes.append(df)
            print(f"Successfully loaded: {file}")
        except Exception as e:
            print(f"Could not load {file}: {e}")
            continue

    # Check if we have at least one dataframe
    if all_dataframes:
        # Concatenate all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)

        # Columns needed for scheduling + professor info + grade columns
        col_names = [
            "Course Number", "Course Name", "Course Area", "Course Available",
            "Course Schedule", "Undergraudate/Graduate", "Instructor", "Subject",
            # Grade distribution columns
            "A", "A-", "A+", "AU", "B", "B-", "B+", "C", "C-", "C+", 
            "D", "D-", "D+", "E", "F"
        ]
        
        # Select only columns that exist in the combined dataframe
        existing_cols = [col for col in col_names if col in combined_df.columns]
        full_df = combined_df[existing_cols].copy()

        # Format Course Number in full_df
        if 'Subject' in full_df.columns and 'Course Number' in full_df.columns:
            full_df['Course Number'] = full_df['Course Number'].astype(str)
            full_df['Course Number'] = full_df.apply(
                lambda row: format_course_number(row['Course Number'], row['Subject']),
                axis=1
            )

        # Create essential dataframe
        # Make sure only existing columns are selected
        essential_cols = [
            "Course Number", "Course Name", "Course Area", "Instructor",
            "Course Available", "Course Schedule", "Undergraudate/Graduate", "Subject"
        ]
        existing_essential_cols = [col for col in essential_cols if col in full_df.columns]
        courses_df = full_df[existing_essential_cols].copy()
        # Get unique departments for dropdown
        departments = sorted(courses_df["Subject"].dropna().unique())
        print(f"Available departments: {departments}")

        # Flag to indicate successful data loading
        data_loaded = True
    else:
        raise Exception("No valid department files were loaded")
        
except Exception as e:
    print(f"Error loading course data: {e}")
    # Create empty dataframes as fallback
    full_df = pd.DataFrame()
    courses_df = pd.DataFrame()
    departments = []
    data_loaded = False

# Helper functions from mychatbot.py
def clean_json_output(output_str):
    """Remove triple-backtick fences and surrounding text from the model's output."""
    output_str = output_str.strip()
    # Remove backticks first
    if output_str.startswith("```"):
        lines = output_str.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        output_str = "\n".join(lines).strip()

    # --- Added: Extract content between first '[' and last ']' ---
    try:
        start_index = output_str.index('[')
        end_index = output_str.rindex(']')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            output_str = output_str[start_index:end_index + 1]
    except ValueError:
        # If '[' or ']' not found, proceed with the string as is (or handle error)
        print(f"Warning: Could not find JSON array brackets in output: {output_str}")
        pass # Keep the potentially non-JSON string for the next step to handle
    # --- End Added ---

    return output_str

def calculate_average_gpa(course_row):
    """Calculate the average GPA for a course based on grade distribution.""" 
    grade_columns = ["A", "A-", "A+", "AU", "B", "B-", "B+", "C", "C-", "C+", "D", "D-", "D+", "E", "F"]
    
    # GPA values for each grade
    gpa_values = {
        "A+": 4.0, "A": 4.0, "A-": 3.7,
        "B+": 3.3, "B": 3.0, "B-": 2.7,
        "C+": 2.3, "C": 2.0, "C-": 1.7,
        "D+": 1.3, "D": 1.0, "D-": 0.7,
        "F": 0.0, "E": 0.0
    }
    
    # Check if course has grade data
    has_grade_data = any(col in course_row for col in grade_columns)
    if not has_grade_data:
        return None
    
    # Calculate average GPA
    total_weighted_gpa = 0
    total_students = 0
    for grade, value in gpa_values.items():
        if grade in course_row and pd.notna(course_row[grade]):
            # Convert grade percentage to decimal (0-1)
            percentage = float(course_row[grade])
            if percentage > 0:
                total_weighted_gpa += value * percentage
                total_students += percentage
    
    if total_students > 0:
        return round(total_weighted_gpa / total_students, 2)
    return None

def generate_schedule(instruction, subject=None, user_info=None, current_schedule=None):
    """Generate a course schedule based on the instruction."""
    # Get available courses only
    available_courses = courses_df[courses_df["Course Available"] == "Y"]
    # Filter by subject if provided
    if subject and subject in available_courses["Subject"].values:
        available_courses = available_courses[available_courses["Subject"] == subject]
        
    # Also include courses from side major if specified
    side_major = user_info.get('side_major', '') if user_info else ''
    if side_major and side_major in courses_df["Subject"].values and side_major != subject:
        side_major_courses = courses_df[(courses_df["Subject"] == side_major) & 
                                        (courses_df["Course Available"] == "Y")]
        available_courses = pd.concat([available_courses, side_major_courses], ignore_index=True)
    
    # Create a copy with only essential columns to avoid memory issues
    essential_columns = ["Course Number", "Course Name", "Course Area", 
                         "Course Schedule", "Undergraudate/Graduate", "Subject", "Instructor"]
    enhanced_courses = available_courses[available_courses.columns.intersection(essential_columns)].copy()
    
    # Calculate and add Average GPA as a column, but don't add all grade columns
    avg_gpas = []
    for _, row in enhanced_courses.iterrows():
        course_num = str(row.get("Course Number", ""))
        # Try to find the course in full_df
        match = full_df[full_df["Course Number"].astype(str) == course_num]
        
        if not match.empty:
            avg_gpa = calculate_average_gpa(match.iloc[0])
            avg_gpas.append(avg_gpa if avg_gpa is not None else "N/A")
        else:
            avg_gpas.append("N/A")
    
    # Add average GPA as a column
    enhanced_courses["Average GPA"] = avg_gpas

    # Convert to CSV - limit the size by not using index=False which can cause extra columns
    available_courses_csv = enhanced_courses.to_csv(index=False)

    # Build user context information
    user_context = ""
    if user_info:
        user_context = f"Student Information:\n"
        if 'name' in user_info and user_info['name']:
            user_context += f"- Name: {user_info['name']}\n"
        if 'year' in user_info and user_info['year']:
            user_context += f"- Year: {user_info['year']}\n"
        if 'major' in user_info and user_info['major']:
            user_context += f"- Primary Major: {user_info['major']}\n"
        if 'side_major' in user_info and user_info['side_major']:
            user_context += f"- Secondary Interest/Minor: {user_info['side_major']}\n"
        if 'interests' in user_info and user_info['interests']:
            user_context += f"- Interests: {user_info['interests']}\n"
        user_context += "\n"
    
    # Include current schedule context if provided
    current_schedule_context = ""
    if current_schedule:
        try:
            simplified_schedule = []
            for course in current_schedule:
                simplified_course = {
                    key: course.get(key, "N/A")
                    for key in ["Course Number", "Course Name", "Course Area"]
                }
                simplified_schedule.append(simplified_course)

            current_schedule_json = json.dumps(simplified_schedule, indent=2)  # Use simplified schedule
            current_schedule_context = (
                "Here is the current schedule that the user has:\n"
                f"{current_schedule_json}\n\n"
                "Please modify this schedule based on the user's new instruction/feedback below.\n"
            )
        except Exception as e:
            print(f"Error converting current schedule to JSON: {e}")
            current_schedule_context = "There was an issue processing the current schedule context.\n"

    prompt = (
        "You are a course scheduler assistant AI.\n"
        f"{user_context}"
        "You should consider user's year, major, and interests when generating the schedule.\n"
        "For example, graduate students want courses above 500 level.\n"
        "If the student has a secondary interest or minor, include 1-2 courses from that area, "
        "but focus primarily on their main major.\n"
        "The dataset includes Average GPA values - higher values (closer to 4.0) indicate easier courses "
        "where students typically receive better grades. Use this information when the user asks for 'easy' courses.\n"
        "Here is the current schedule:\n"
        f"{current_schedule_context}"
        "Below is a dataset of courses in CSV format. Only consider the courses that are available next semester and please only use the data from the coursedata:\n\n"
        f"{available_courses_csv}\n\n"
        "Using ONLY these available courses, please generate a course schedule that meets the following instruction:\n"
        f"{instruction}\n\n"
        "Make sure the schedule avoids any time conflicts (if possible, based on the 'Course Schedule' column). "
        "Return only the course schedule as a valid JSON array. Each element in the array should be a JSON object with the following keys: "
        "'Course Number', 'Course Name', 'Course Area', 'Course Schedule', 'Undergraudate/Graduate'."
    )
    
    # Reset conversation for a fresh context
    chatbot.reset_conversation()
    chatbot.update_system_prompt("You are a course scheduler assistant AI.")
    
    # Generate response
    response = chatbot.generate_response(prompt)
    print(response)
    return response

def parse_schedule_json(raw_output):
    """Parse the schedule JSON from the raw output.""" 
    cleaned = clean_json_output(raw_output)
    try:
        schedule_list = json.loads(cleaned)
        # Ensure it's a list
        if not isinstance(schedule_list, list):
            return []
        
        # Add Subject if missing, using the formatted Course Number for lookup
        for course in schedule_list:
            if "Course Number" in course and "Subject" not in course:
                course_num = course["Course Number"]
                match = full_df[full_df["Course Number"] == course_num]
                if not match.empty and "Subject" in match.columns:
                    course["Subject"] = match.iloc[0]["Subject"]
        return schedule_list
    except json.JSONDecodeError:
        return []

def generate_grade_distribution(schedule_list):
    """Generate grade distribution data for the given schedule.""" 
    grade_columns = ["A", "A-", "A+", "AU", "B", "B-", "B+", "C", "C-", "C+", "D", "D-", "D+", "E", "F"]
    distributions = {}
    # GPA values for each grade
    gpa_values = {
        "A+": 4.0, "A": 4.0, "A-": 3.7,
        "B+": 3.3, "B": 3.0, "B-": 2.7,
        "C+": 2.3, "C": 2.0, "C-": 1.7,
        "D+": 1.3, "D": 1.0, "D-": 0.7,
        "F": 0.0, "E": 0.0
    }
    for course in schedule_list:
        course_num = course.get("Course Number", "")
        row_match = full_df[full_df["Course Number"] == course_num]
        
        if row_match.empty:
            distributions[course_num] = "Course not found"
        else:
            row = row_match.iloc[0]
            # Get individual grade distributions for columns that exist
            distribution = {}
            for g in grade_columns:
                if g in row:
                    distribution[g] = float(row[g]) * 100 if pd.notna(row[g]) else 0
            # Calculate combined letter grade distributions
            combined_dist = {
                "A": sum(distribution.get(g, 0) for g in ["A+", "A", "A-"]),
                "B": sum(distribution.get(g, 0) for g in ["B+", "B", "B-"]),
                "C": sum(distribution.get(g, 0) for g in ["C+", "C", "C-"]),
                "D": sum(distribution.get(g, 0) for g in ["D+", "D", "D-"]),
                "F": distribution.get("F", 0) + distribution.get("E", 0)
            }
            
            # Calculate average GPA
            total_weighted_gpa = 0
            total_students = 0
            for grade, percentage in distribution.items():
                if grade != "AU" and grade in gpa_values:  # Skip AU (audit) grades
                    total_weighted_gpa += gpa_values[grade] * percentage
                    total_students += percentage
            
            avg_gpa = total_weighted_gpa / total_students if total_students > 0 else 0
            
            # Store all data
            distributions[course_num] = {
                "detailed": distribution,
                "combined": combined_dist,
                "avg_gpa": round(avg_gpa, 2)
            }
    
    return distributions

def enrich_schedule_with_grades(schedule_list):
    """Add grade columns to each course in the schedule.""" 
    grade_columns = ["A", "A-", "A+", "AU", "B", "B-", "B+", "C", "C-", "C+", "D", "D-", "D+", "E", "F"]
    enriched_schedule = []
    
    for course in schedule_list:
        # Create an enriched course object
        enriched_course = course.copy()
        
        # Find the course in the full dataframe
        course_num = course.get("Course Number", "")
        row_match = full_df[full_df["Course Number"] == course_num]
        
        # Add grade data if the course is found
        if not row_match.empty:
            row = row_match.iloc[0]
            for g in grade_columns:
                if g in row:
                    enriched_course[g] = float(row[g]) if pd.notna(row[g]) else 0
        
        enriched_schedule.append(enriched_course)
    
    return enriched_schedule

def ask_question_about_courses(question, schedule_list, subject=None, side_major=None):
    """Ask a question about the courses or schedule."""
    # Get relevant dataset for the subject and side_major
    dfs_to_concat = []
    if subject and subject in full_df["Subject"].values:
        dfs_to_concat.append(full_df[full_df["Subject"] == subject])

    if side_major and side_major in full_df["Subject"].values and side_major != subject:
        dfs_to_concat.append(full_df[full_df["Subject"] == side_major])

    # Concatenate the dataframes if any were added
    if dfs_to_concat:
        relevant_df = pd.concat(dfs_to_concat, ignore_index=True)
    else:
        # If no specific subjects were valid or provided, use the full dataset
        relevant_df = full_df.copy()

    # Sort by 'Instructor' placing NaNs last to prioritize keeping rows with instructor names.
    if 'Instructor' in relevant_df.columns:
        relevant_df = relevant_df.sort_values(by='Instructor', na_position='last', ascending=True)
    
    # Drop duplicates based on both Subject and Course Number, keeping the first occurrence.
    relevant_df = relevant_df.drop_duplicates(subset=["Subject", "Course Number"], keep='first')
    
    # Optional: Sort back by original index order and reset index for consistency.
    relevant_df = relevant_df.sort_index().reset_index(drop=True)

    dataset_csv = relevant_df.to_csv(index=False)
    schedule_json_str = json.dumps(schedule_list, indent=2)

    prompt = (
        "Here is the dataset of courses for the student's major and minor/secondary interest:\n"
        f"{dataset_csv}\n\n"
        "Here is the current selected schedule:\n"
        f"{schedule_json_str}\n\n"
        "User question:\n"
        f"{question}\n\n"
        "Please answer using the dataset or the schedule as needed. Make your response conversational, friendly and helpful."
    )

    # Reset conversation for a fresh context
    chatbot.reset_conversation()
    chatbot.update_system_prompt("You are a helpful course assistant.")

    # Generate response
    response = chatbot.generate_response(prompt)
    return response

def check_schedule_conflicts(schedule_list):
    """Check for time conflicts in the schedule.""" 
    time_slots = {}
    conflicts = []
    
    for course in schedule_list:
        schedule = course.get("Course Schedule", "")
        if schedule:
            if schedule in time_slots:
                conflicts.append({
                    "course1": course["Course Number"],
                    "course2": time_slots[schedule]["Course Number"],
                    "time": schedule
                })
            else:
                time_slots[schedule] = course
    
    return conflicts

def resolve_schedule_conflicts(instruction, schedule_list, subject=None, user_info=None):
    """Try to generate a new schedule without conflicts.""" 
    if not schedule_list:
        return []
    
    conflict_resolution = f"{instruction} and make sure there are no time conflicts."
    raw_schedule_output = generate_schedule(conflict_resolution, subject, user_info)
    new_schedule = parse_schedule_json(raw_schedule_output)
    return new_schedule

# Flask routes
@app.route('/')
def index():
    """Render the main page.""" 
    # Initialize or reset the session data
    session['student_info'] = {}
    session['schedule'] = []
    session['instruction'] = ""
    return render_template('course_scheduler.html', data_loaded=data_loaded, departments=departments)

@app.route('/save_student_info', methods=['POST'])
def save_student_info():
    """Save student information to the session.""" 
    try:
        student_info = {
            'name': request.form.get('name', ''),
            'year': request.form.get('year', ''),
            'major': request.form.get('major', ''),
            'side_major': request.form.get('side_major', ''),
            'interests': request.form.get('interests', '')
        }
        session['student_info'] = student_info
        
        # Store the subject for later use
        session['subject'] = student_info['major']
        # Store side major if provided
        if student_info['side_major']:
            session['side_major'] = student_info['side_major']
        
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/generate_schedule', methods=['POST'])
def create_schedule():
    """Generate a course schedule based on user instruction.""" 
    if not data_loaded:
        return jsonify({
            'status': 'error',
            'message': 'Course data not loaded. Please check if the department files exist.'
        })
    
    try:
        instruction = request.form.get('instruction', '')
        if not instruction:
            return jsonify({
                'status': 'error',
                'message': 'Please provide instructions for your schedule.'
            })
        
        # Store the instruction in session
        session['instruction'] = instruction
        
        # Get the subject and student info from session
        subject = session.get('subject')
        user_info = session.get('student_info', {})
        
        # Generate the schedule with user information
        raw_schedule_output = generate_schedule(instruction, subject, user_info)
        schedule_list = parse_schedule_json(raw_schedule_output)
        
        if not schedule_list:
            return jsonify({
                'status': 'error',
                'message': 'Could not generate a valid schedule. Please try different instructions.'
            })
        
        # Enrich the schedule with grade data
        enriched_schedule = enrich_schedule_with_grades(schedule_list)
        
        # Store the schedule in session
        session['schedule'] = enriched_schedule
        
        # Check for conflicts
        conflicts = check_schedule_conflicts(enriched_schedule)
        
        # Get list of course areas for the summary
        areas = set()
        for course in enriched_schedule:
            if "Course Area" in course and course["Course Area"]:
                areas.add(course["Course Area"])
        
        return jsonify({
            'status': 'success',
            'schedule': enriched_schedule,
            'courseCount': len(enriched_schedule),
            'conflicts': conflicts,
            'areas': list(areas)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error generating schedule: {str(e)}'
        })

@app.route('/get_grade_distribution', methods=['POST'])
def get_grades():
    """Get grade distribution for the current schedule.""" 
    try:
        schedule_list = session.get('schedule', [])
        if not schedule_list:
            return jsonify({
                'status': 'error',
                'message': 'No schedule available. Please generate a schedule first.'
            })
        
        grade_dist = generate_grade_distribution(schedule_list)
        
        # Calculate insights about GPAs
        gpa_insights = []
        course_gpas = []
        
        for course_num, dist_data in grade_dist.items():
            if isinstance(dist_data, dict) and "avg_gpa" in dist_data:
                course_gpas.append((course_num, dist_data["avg_gpa"]))
        
        if course_gpas:
            course_gpas.sort(key=lambda x: x[1], reverse=True)
            highest_gpa = course_gpas[0]
            gpa_insights.append(f"Course {highest_gpa[0]} has the highest average GPA of {highest_gpa[1]:.2f}.")
            
            if len(course_gpas) > 1:
                lowest_gpa = course_gpas[-1]
                gpa_insights.append(f"Course {lowest_gpa[0]} has the lowest average GPA of {lowest_gpa[1]:.2f}.")
        
        # Calculate A-grade percentages for insights
        a_percentages = []
        for course_num, dist_data in grade_dist.items():
            if isinstance(dist_data, dict) and "combined" in dist_data:
                a_grade = dist_data["combined"]["A"]
                if a_grade > 0:
                    a_percentages.append((course_num, a_grade))
        
        insights = gpa_insights
        if a_percentages:
            a_percentages.sort(key=lambda x: x[1], reverse=True)
            highest_a = a_percentages[0]
            insights.append(f"Course {highest_a[0]} has the highest percentage of A grades ({highest_a[1]:.1f}%).")
            
            if len(a_percentages) > 1:
                lowest_a = a_percentages[-1]
                insights.append(f"Course {lowest_a[0]} has the lowest percentage of A grades ({lowest_a[1]:.1f}%).")
        
        return jsonify({
            'status': 'success',
            'grade_distribution': grade_dist,
            'insights': insights
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error getting grade distribution: {str(e)}'
        })

@app.route('/ask_question', methods=['POST'])
def ask_question():
    """Ask a question about courses or the schedule."""
    try:
        question = request.form.get('question', '')
        if not question:
            return jsonify({
                'status': 'error',
                'message': 'Please provide a question to ask.'
            })
        
        schedule_list = session.get('schedule', [])
        subject = session.get('subject')
        side_major = session.get('side_major')
        
        answer = ask_question_about_courses(question, schedule_list, subject, side_major)
        
        return jsonify({
            'status': 'success',
            'answer': answer
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error processing question: {str(e)}'
        })

@app.route('/resolve_conflicts', methods=['POST'])
def resolve_conflicts():
    """Try to resolve schedule conflicts.""" 
    try:
        schedule_list = session.get('schedule', [])
        instruction = session.get('instruction', '')
        subject = session.get('subject')
        user_info = session.get('student_info', {})
        
        if not schedule_list or not instruction:
            return jsonify({
                'status': 'error',
                'message': 'No schedule or instruction available.'
            })
        
        new_schedule = resolve_schedule_conflicts(instruction, schedule_list, subject, user_info)
        if not new_schedule:
            return jsonify({
                'status': 'error',
                'message': 'Could not resolve all conflicts. Try manually removing some courses.'
            })
        
        # Enrich the new schedule with grade data
        enriched_schedule = enrich_schedule_with_grades(new_schedule)
        
        # Update the session with the new enriched schedule
        session['schedule'] = enriched_schedule
        
        # Check if there are still conflicts
        conflicts = check_schedule_conflicts(enriched_schedule)
        
        return jsonify({
            'status': 'success',
            'schedule': enriched_schedule,
            'conflicts': conflicts,
            'resolved': len(conflicts) == 0
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error resolving conflicts: {str(e)}'
        })

@app.route('/update_schedule', methods=['POST'])
def update_schedule():
    """Update the schedule with new instructions.""" 
    try:
        feedback = request.form.get('feedback', '')
        instruction = session.get('instruction', '')
        subject = session.get('subject')
        user_info = session.get('student_info', {})
        current_schedule_list = session.get('schedule', [])
        
        if not instruction:
            return jsonify({
                'status': 'error',
                'message': 'No previous instruction found. Please generate a schedule first.'
            })
        
        updated_instruction = f"Original request: {instruction}\nFeedback/Update: {feedback}"
        raw_schedule_output = generate_schedule(updated_instruction, subject, user_info, current_schedule=current_schedule_list)
        new_schedule = parse_schedule_json(raw_schedule_output)
        
        if not new_schedule:
            return jsonify({
                'status': 'error',
                'message': 'Could not update schedule with those requirements.'
            })
        
        # Enrich the new schedule with grade data
        enriched_schedule = enrich_schedule_with_grades(new_schedule)
        
        # Update session data
        session['schedule'] = enriched_schedule
        session['instruction'] = updated_instruction
        
        # Check for conflicts
        conflicts = check_schedule_conflicts(enriched_schedule)
        
        return jsonify({
            'status': 'success',
            'schedule': enriched_schedule,
            'conflicts': conflicts
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error updating schedule: {str(e)}'
        })

if __name__ == '__main__':
    app.run(debug=True)