import pandas as pd

def get_expected_file_split_stats(df_master, df_c, df_r):
    print(f"+{'-'*20}+")
    print(f"| Total    : {len(df_master):<5}  |")
    print(f"| Complete : {len(df_c):<5}  |")
    print(f"| Reprocess: {len(df_r):<5}   |")
    print(f"|{'-'*20}|")
    print(f"| ToDo     : {len(df_master) - (len(df_c)+len(df_r)):<5}   |")
    print(f"+{'-'*20}+")

def get_chunk_processing_stats(df: pd.DataFrame, CHUNK_SIZE=200):
    """Processes DataFrame in chunks, printing progress and conversion rate if a null-only chunk is found."""
    total_chunks = len(df) // CHUNK_SIZE + (1 if len(df) % CHUNK_SIZE != 0 else 0)
    processed_chunks = 0
    non_null_rows = 0
    start_index, end_index = 0, len(df)
    
    for i in range(start_index, end_index, CHUNK_SIZE):
        chunk = df.iloc[i:i+CHUNK_SIZE]

        non_null_count = chunk['rawText'].notnull().sum()
        non_null_rows += non_null_count

        # Check if the entire chunk's rawText column is null
        if chunk['rawText'].isnull().all():
            processed_percentage = (processed_chunks / total_chunks) * 100
            conversion_rate = (non_null_rows / (end_index - start_index)) * 100
            
            # Print results in a more readable format
            print(f"\n{'Progress Summary':^80}")
            print(f"{'Processed Chunks':<20}{'Total Chunks':<20}{'Percentage (%)':<20}{'Overall Conversion Rate (%)':<20}")
            print(f"{processed_chunks:<20}{total_chunks:<20}{processed_percentage:<20.2f}{conversion_rate:<20.2f}")

            preview = pd.concat([df.iloc[max(0, i-CHUNK_SIZE):i].tail(5), df.iloc[i:i+CHUNK_SIZE].head(5)])
            return preview

        processed_chunks += 1
    return "COMPLETE"

def get_dataset_stats(file_path:str):
    df = pd.read_excel(file_path)
    
    total_records = len(df)
    df_isProfessor = df[df['isProfessor'] == True]
    df_professorWithDept_textual = df_isProfessor[df_isProfessor['department_textual'] != 'MISSING']
    df_professorWithDept_keyword = df_isProfessor[df_isProfessor['department_keyword'] != 'MISSING']
    df_professorWithDept_either = df_isProfessor[
        (df_isProfessor['department_textual'] != 'MISSING') | 
        (df_isProfessor['department_keyword'] != 'MISSING')
    ]

    # Calculating Stats
    num_professors = len(df_isProfessor)
    num_professors_with_dept_textual = len(df_professorWithDept_textual)
    num_professors_with_dept_keyword = len(df_professorWithDept_keyword)
    num_professors_with_dept_either = len(df_professorWithDept_either)
    percent_with_dept = (num_professors_with_dept_either / num_professors) * 100 if num_professors else 0

    # Output stats
    print(f"\n ________________________________________________ ")
    print(f"|{'STATS FOR: ' + file_path:^48}|")
    print(f"|________________________________________________|")
    print(f"|============= Professor Statistics =============|")
    print(f"{'|Total Number of Records:':<41} {total_records:<7}|")
    print(f"{'|Number of Professors:':<41} {num_professors:<7}|")
    print(f"{'|Professors with Department (textual):':<41} {num_professors_with_dept_textual:<7}|")
    print(f"{'|Professors with Department (keyword):':<41} {(num_professors_with_dept_keyword):<7}|")
    print(f"{'|Number of Professors without:':<41} {(num_professors - num_professors_with_dept_either):<7}|")
    print(f"|________________________________________________|")
    print(f"|=============== Conversion Rates ===============|")
    print(f"{'|Professor Identification Rate (%):':<41} {((num_professors / total_records) * 100):<7.2f}|")
    print(f"{'|Department Extraction Rate (coverage %):':<41} {percent_with_dept:<7.2f}|")
    print(f"{'|Department Coverage Gap (slippage %):':<41} {(100 - percent_with_dept):<7.2f}|")
    print(f"|________________________________________________|")