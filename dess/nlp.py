import re
import pandas as pd
import pickle

# ------------------------------------------------------------------------------
# Config

# criteria associated with dummy variables
CRITERIA_FLAGS = {
    'isProfessor': ["professor", "faculty"],
    'isInstructor': ["instructor", "educator", "adjunct", "lecturer", "teacher"],
    'isEmeritus': ["emeritus", "emerita", "emiritus", "emirita"],
    'isAssistantProf': ["assistant"],
    'isAssociateProf': ["associate"],
    'isFullProf': ["full"],
    'isClinicalProf': ["clinical"],
    'isResearcher': ["research", "citations", "examine", "investigate"],
    'isRetired': ["emiritus", "emerita", "retired", "passed away", "memorial", "obituary", 
                  "death", "tribute", "funeral", "condolences"],
}

# Patterns to match department names, ordered by priority
DEPARTMENT_PATTERNS = {
    # Primary patterns - indicate professor role (sets isProfessor2=True)
    'primary': [
        r'professor in the (?:dept|department) of(?: the| public)? ([A-Za-z]+)',
        r'(?:of|in)(?: the| public)? ([A-Za-z]+) (?:dept|department)',
        r'(?:in the )?(?:dept|department)(?:s|.)? of(?: the|.| public)? ([A-Za-z]+)',
        r'the ([A-Za-z]+) department',
        r'professor (?:of|in)(?: the)? ([A-Za-z]+)',
        r'chair in(?: the)? ([A-Za-z]+)',
        r'professor emerit(?:us|a) of(?: the| public)? ([A-Za-z]+)',
        r'faculty of(?: the)? ([A-Za-z]+)',
        r'(?:of|in) the ([A-Za-z]+) ([A-Za-z]+) (?:dept|department)'
    ],
        
    # Backup patterns - contextual department mentions
    'backup': [
        r'(?:a|an) ([A-Za-z]+) professor',
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
        r'ph\.?d\.?\s*(?:degree\s+)?(?:in|of|from)?\s*([A-Za-z]+)',
        r'is (?:a|an) ([A-Za-z]+) professor',
        r'professor, ([A-Za-z]+)'
    ]
}

# Words to ignore if this is the department that's extracted — minimize false positives
IGNORE_TERMS = ['the', 'department','assistant','associate','full','special','university','adjunct',
                'school','senior','college','emeritus', 'degree', 'current', 'phone', 'faculty', 
                'dept', 'in', 'research', 'professor', 'specialty']

# Path to the file containing the whitelist of keywords for department extraction
KEYWORD_WHITELIST_FILE_PATH = "storage/department-whitelist.pkl"
# ------------------------------------------------------------------------------

def extract_department_information(df: pd.DataFrame):
    """Populates the isFaculty and department columns in the DataFrame."""
    df[['isProfessor', 'isInstructor', 'isEmeritus', 'isAssistantProf', 'isAssociateProf', 
        'isFullProf', 'isClinicalProf', 'isResearcher', 'isRetired', 'teaching_intensity', 'department_textual',
        'isPrimaryPattern', 'department_keyword', 'keyword_precision']] =  df.apply(
        lambda row: populate_faculty_columns(row['rawText']),
        axis=1,
        result_type='expand'
    )

def populate_faculty_columns(rawText: list[str]):
    flags = populate_dummy_variables(rawText)
    department_textual, isPrimaryPattern, department_keyword, keyword_precision  = populate_department_variables(rawText)
    return  (*flags, department_textual, isPrimaryPattern, department_keyword, keyword_precision)

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
    department_textual, isPrimaryPattern, department_keyword, keyword_precision = "MISSING", -1, "MISSING", 0
    
    if rawText is None:
        return department_textual, isPrimaryPattern, department_keyword, keyword_precision
    
    department_textual, isPrimaryPattern = _extract_department_regex(rawText)
    department_keyword, keyword_precision = _extract_department_fuzzy_match(rawText)

    return department_textual, isPrimaryPattern, department_keyword, keyword_precision

def _extract_department_regex(rawText):
    # Try primary patterns first
    for text in rawText:
        for pattern in DEPARTMENT_PATTERNS['primary']:
            if match := re.search(pattern, text, re.IGNORECASE):
                department_textual = match.group(1).strip().lower()
                
                # Skip terms in the ignore list (to avoid false positives)
                if department_textual in IGNORE_TERMS:
                    continue
                return department_textual, 1

    # Fall back to secondary patterns            
    for text in rawText:
        for pattern in DEPARTMENT_PATTERNS['backup']:
            if match := re.search(pattern, text, re.IGNORECASE):
                department_textual = match.group(1).strip().lower()
                
                # Skip terms in the ignore list (to avoid false positives)
                if department_textual in IGNORE_TERMS:
                    continue
                return department_textual, 0
            
    return "MISSING", -1

def create_keyword_dict_file(excel_file_path):
    df_keywords = pd.read_excel(excel_file_path)
    df_keywords = df_keywords.dropna(subset=['department_keyword'])

    keywords = df_keywords['department_keyword'].str.lower()
    precision = df_keywords['Precision Level ']

    keyword_dict = {1: [], 2: [], 3: []}
    for keyword, prec in zip(keywords, precision):
        if prec in keyword_dict:
            keyword_dict[prec].append(keyword)

    with open(KEYWORD_WHITELIST_FILE_PATH, 'wb') as f:
        pickle.dump(keyword_dict, f)

def _load_department_names(file_path):
    with open(file_path, 'rb') as f:
        department_names = pickle.load(f)
    return department_names

def _extract_department_fuzzy_match(rawText):
    DEPARTMENT_WHITELIST = _load_department_names(KEYWORD_WHITELIST_FILE_PATH)

    for i in range(1, 4):
        for text in rawText:
            for department in DEPARTMENT_WHITELIST[i]:
                if department in text.lower(): return department, i

    return "MISSING", -1

def _count_teaching_intensity(text: str) -> int:
    """Counts the number of times the word teach appears in the text using regex."""
    matches = re.findall(r'\bteach\w*', text, re.IGNORECASE)
    return len(matches)
    
def _lookup_criteria(text: str, criteria: list[str]) -> bool:
    for criterion in criteria:
        if criterion in text.lower():
            return True
    return False

if __name__ == "__main__":
    df_recent = pd.read_excel('storage/googleApiSearch_test_recent.xlsx')
    df_recent=df_recent.drop(columns='Unnamed: 0')
    df_recent = df_recent.dropna(subset=['rawText'])
    extract_department_information(df_recent)