import re
import pandas as pd
import pickle

# ------------------------------------------------------------------------------
# Config

# criteria associated with dummy variables
CRITERIA_FLAGS = {
    'isProfessor': ["professor", "faculty"],
    'isInstructor': ["instructor", "educator", "adjunct", "lecturer", "professor of teaching"],
    'isEmeritus': ["emiritus", "emerita"],
    'isAssistantProf': ["assistant"],
    'isAssociateProf': ["associate"],
    'isFullProf': ["full"],
    'isClinicalProf': ["clinical"],
    'isResearcher': ["research", "citations", "examine", "investigate"]
}

# Patterns to match department names, ordered by priority
DEPARTMENT_PATTERNS = {
    # Primary patterns - indicate professor role (sets isProfessor2=True)
    'primary': [
        r'professor in the department of(?: the| public)? ([A-Za-z]+)',
        r'(?:of|in)(?: the| public)? ([A-Za-z]+) department',
        r'(?:in the )?department(?:s|.)? of(?: the|.| public)? ([A-Za-z]+)',
        r'department: ([A-Za-z]+)',
        r'professor (?:of|in)(?: the)? ([A-Za-z]+)',
        r'chair in(?: the)? ([A-Za-z]+)',
        r'professor emerit(?:us|a) of(?: the| public)? ([A-Za-z]+)',
        r'faculty of(?: the)? ([A-Za-z]+)',
        r'(?:of|in) the ([A-Za-z]+) ([A-Za-z]+) department',
        r'professor, ([A-Za-z]+)'
    ],
        
    # Backup patterns - contextual department mentions
    'backup': [
        r'book on(?: the)? ([A-Za-z]+)',
        r'in the area of(?: the)? ([A-Za-z]+)',
        r'research(?: primarily)? focused on(?: the)? ([A-Za-z]+)',
        r'(?:area of|research|areas of|) interest(?:s)(?::|.) ([A-Za-z]+)',
        r'research focus(?:es)? on(?: the)? ([A-Za-z]+)',
        r'research interest(?:s)?: ([A-Za-z]+)',
        r'expert in(?: the)? ([A-Za-z]+)',
        r'leader in(?: the)? ([A-Za-z]+)',
        r'(?:school|college) of(?: the| public)? ([A-Za-z]+)',
        r'center for(?: the)? ([A-Za-z]+)',
        r'\bph\.?d\.?\s*(?:in|of|from)?\s*([A-Za-z]+)',
        r'is (?:a|an) ([A-Za-z]+) professor'
    ]
}

# Words to ignore if this is the department that's extracted — minimize false positives
IGNORE_TERMS = ['the', 'department','assistant','associate','full','special','university','adjunct','school','senior','college','emeritus']

# ------------------------------------------------------------------------------

def extract_department_information(df: pd.DataFrame):
    """Populates the isFaculty and department columns in the DataFrame."""
    df[['isProfessor', 'isInstructor', 'isEmeritus', 'isAssistantProf', 'isAssociateProf', 
        'isFullProf', 'isClinicalProf', 'isResearcher', 'teaching_intensity', 'isProfessor2', 
        'department_textual', 'department_keyword']] =  df.apply(
        lambda row: populate_faculty_columns(row['rawText']),
        axis=1,
        result_type='expand'
    )

def populate_faculty_columns(rawText: list[str]):
    flags = populate_dummy_variables(rawText)
    isProfessor2, department_textual, department_keyword  = populate_department_variables(rawText)
    return  (*flags, isProfessor2, department_textual, department_keyword)

def populate_dummy_variables(rawText: list[str]) -> str:
    if rawText is None:
        return False, False, False, False, False, False, False, False, 0

    flags = {key: False for key in CRITERIA_FLAGS.keys()}
    teaching_intensity = 0

    for text in rawText:
        teaching_intensity += _count_teaching_intensity(text)
        for flag, criteria in CRITERIA_FLAGS.items():
            if flags[flag] == False and _lookup_criteria(text, criteria):
                flags[flag] = True

    return  tuple(flags.values()) + (teaching_intensity,)

def populate_department_variables(rawText):
    """
    Uses regex to extract department and populates all 
    department-related variables.
    """
    department_textual, department_keyword, isProfessor2 = "MISSING", "MISSING", False
    
    if rawText is None:
        return isProfessor2, department_textual, department_keyword
    
    department_textual, isProfessor2 = _extract_department_regex(rawText)
    department_keyword = _extract_department_fuzzy_match(rawText)

    return isProfessor2, department_textual, department_keyword

def _extract_department_regex(rawText):
    # Try primary patterns first
    for text in rawText:
        for pattern in DEPARTMENT_PATTERNS['primary']:
            if match := re.search(pattern, text, re.IGNORECASE):
                department_textual = match.group(1).strip().lower()
                
                # Skip terms in the ignore list (to avoid false positives)
                if department_textual in IGNORE_TERMS:
                    continue
                return department_textual, True

    # Fall back to secondary patterns            
    for text in rawText:
        for pattern in DEPARTMENT_PATTERNS['backup']:
            if match := re.search(pattern, text, re.IGNORECASE):
                department_textual = match.group(1).strip().lower()
                
                # Skip terms in the ignore list (to avoid false positives)
                if department_textual in IGNORE_TERMS:
                    continue
                return department_textual, False
            
    return "MISSING", False # Fallback if no match

def _load_department_names(file_path):
    with open(file_path, 'rb') as f:
        department_names = pickle.load(f)
    return department_names

def _extract_department_fuzzy_match(rawText):
    DEPARTMENT_WHITELIST = _load_department_names("storage/department-whitelist.pkl")

    for text in rawText:
        for department in DEPARTMENT_WHITELIST:
            if department in text.lower(): return department
    return "MISSING"

def _count_teaching_intensity(text: str) -> int:
    """Counts the number of times the word teach appears in the text using regex."""
    matches = re.findall(r'\bteach\w*', text, re.IGNORECASE)
    return len(matches)
    
def _lookup_criteria(text: str, criteria: list[str]) -> bool:
    for criterion in criteria:
        if criterion in text.lower():
            return True
    return False